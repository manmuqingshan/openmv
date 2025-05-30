/*
 * This file is part of the MicroPython project, http://micropython.org/
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2013, 2024 Damien P. George
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <stdlib.h>
#include "py/mphal.h"
#include "py/runtime.h"
#include "shared/runtime/interrupt_char.h"
#include "pendsv.h"
#include CMSIS_MCU_H

// This variable is used to save the exception object between a ctrl-C and the
// PENDSV call that actually raises the exception.  It must be non-static
// otherwise gcc-5 optimises it away.  It can point to the heap but is not
// traced by GC.  This is okay because we only ever set it to
// mp_kbd_exception which is in the root-pointer set.
void *pendsv_object;
static int pendsv_lock;

#if defined(PENDSV_DISPATCH_NUM_SLOTS)
uint32_t pendsv_dispatch_active;
pendsv_dispatch_t pendsv_dispatch_table[PENDSV_DISPATCH_NUM_SLOTS];
#endif

void pendsv_init(void) {
    pendsv_object = NULL;
    MP_STATE_MAIN_THREAD(mp_pending_exception) = MP_OBJ_NULL;
    #if defined(PENDSV_DISPATCH_NUM_SLOTS)
    pendsv_dispatch_active = false;
    #endif
    // Set PendSV to lowest priority.
    NVIC_SetPriority(PendSV_IRQn, IRQ_PRI_PENDSV);
}

// Call this function to raise a pending exception during an interrupt.
// It will first try to raise the exception "softly" by setting the
// mp_pending_exception variable and hoping that the VM will notice it.
// If this function is called a second time (ie with the mp_pending_exception
// variable already set) then it will force the exception by using the hardware
// PENDSV feature.  This will wait until all interrupts are finished then raise
// the given exception object using nlr_jump in the context of the top-level
// thread.
void pendsv_kbd_intr(void) {
    if (MP_STATE_MAIN_THREAD(mp_pending_exception) == MP_OBJ_NULL) {
        mp_sched_keyboard_interrupt();
    } else {
        MP_STATE_MAIN_THREAD(mp_pending_exception) = MP_OBJ_NULL;
        pendsv_object = &MP_STATE_VM(mp_kbd_exception);
        SCB->ICSR = SCB_ICSR_PENDSVSET_Msk;
    }
}

void pendsv_suspend(void) {
    pendsv_lock++;
}

void pendsv_resume(void) {
    pendsv_lock--;
    assert(pendsv_lock >= 0);
    #if defined(PENDSV_DISPATCH_NUM_SLOTS)
    // Run pendsv if needed.  Find an entry with a dispatch and call pendsv dispatch
    // with it.  If pendsv runs it will service all slots.
    int count = PENDSV_DISPATCH_NUM_SLOTS;
    while (count--) {
        if (pendsv_dispatch_table[count]) {
            pendsv_schedule_dispatch(count, pendsv_dispatch_table[count]);
            break;
        }
    }
    #endif
}

#if defined(PENDSV_DISPATCH_NUM_SLOTS)
void pendsv_schedule_dispatch(size_t slot, pendsv_dispatch_t f) {
    if (slot < PENDSV_DISPATCH_NUM_SLOTS) {
        pendsv_dispatch_table[slot] = f;
        pendsv_dispatch_active = true;
        SCB->ICSR = SCB_ICSR_PENDSVSET_Msk;
    }
}

void pendsv_dispatch_handler(void) {
    for (size_t i = 0; i < PENDSV_DISPATCH_NUM_SLOTS; ++i) {
        if (pendsv_dispatch_table[i] != NULL) {
            pendsv_dispatch_t f = pendsv_dispatch_table[i];
            pendsv_dispatch_table[i] = NULL;
            f();
        }
    }
}
#endif

__attribute__((naked)) void PendSV_Handler(void) {
    // Handle a PendSV interrupt
    //
    // For the case of an asynchronous exception, re-jig the
    // stack so that when we return from this interrupt handler
    // it returns instead to nlr_jump with argument pendsv_object
    //
    // on entry to this (naked) function, stack has the following layout:
    //   sp[6]: pc=r15
    //   sp[5]: lr=r14
    //   sp[4]: r12
    //   sp[3]: r3
    //   sp[2]: r2
    //   sp[1]: r1
    //   sp[0]: r0

    __asm volatile (
        #if defined(PENDSV_DISPATCH_NUM_SLOTS)
        // Check if there are any pending calls to dispatch to
        "ldr r1, pendsv_dispatch_active_ptr\n"
        "ldr r0, [r1]\n"
        "cmp r0, #0\n"
        "beq .no_dispatch\n"
        "mov r2, #0\n"
        "str r2, [r1]\n"                // clear pendsv_dispatch_active
        "b pendsv_dispatch_handler\n"   // jump to the handler
        ".no_dispatch:\n"
        #endif

        // Check if there is an active object to throw via nlr_jump
        "ldr r1, pendsv_object_ptr\n"
        "ldr r0, [r1]\n"
        "cmp r0, #0\n"
        "beq .no_obj\n"
        // Modify stacked XPSR to make sure
        // possible LDM/STM progress is cleared.
        #if __ARM_ARCH >= 7
        "mov r2, #0x01000000 \n"
        #else
        "mov r2,  #1 \n"
        "lsl r2,  #24\n"
        #endif
        "str r2, [sp, #28] \n"
        "str r0, [sp, #0]\n"            // store to r0 on stack
        "mov r0, #0\n"
        "str r0, [r1]\n"                // clear pendsv_object
        "ldr r0, nlr_jump_ptr\n"
        "str r0, [sp, #24]\n"           // store to pc on stack
        "bx lr\n"                       // return from interrupt; will return to nlr_jump
        ".no_obj:\n"                    // pendsv_object==NULL

        #if MICROPY_PY_THREAD && __ARM_ARCH >= 7
        // Do a thread context switch
        "push {r4-r11, lr}\n"
        "vpush {s16-s31}\n"
        "mrs r5, primask\n"             // save PRIMASK in r5
        "cpsid i\n"                     // disable interrupts while we change stacks
        "mov r0, sp\n"                  // pass sp to save
        "mov r4, lr\n"                  // save lr because we are making a call
        "bl pyb_thread_next\n"          // get next thread to execute
        "mov lr, r4\n"                  // restore lr
        "mov sp, r0\n"                  // switch stacks
        "msr primask, r5\n"             // re-enable interrupts
        "vpop {s16-s31}\n"
        "pop {r4-r11, lr}\n"
        "bx lr\n"                       // return from interrupt; will return to new thread
        #else
        // Spurious pendsv, just return
        "bx lr\n"
        #endif

        // Data
        ".align 2\n"
        #if defined(PENDSV_DISPATCH_NUM_SLOTS)
        "pendsv_dispatch_active_ptr: .word pendsv_dispatch_active\n"
        #endif
        "pendsv_object_ptr: .word pendsv_object\n"
        "nlr_jump_ptr: .word nlr_jump\n"
        );
}
