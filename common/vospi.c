/*
 * SPDX-License-Identifier: MIT
 *
 * Copyright (C) 2023-2024 OpenMV, LLC.
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
 *
 * VOSPI driver.
 */
#include "omv_boardconfig.h"
#if OMV_ENABLE_VOSPI || OMV_LEPTON_ENABLE

#include <stdint.h>
#include <string.h>
#include "py/mphal.h"

#include "vospi.h"
#include "crc16.h"
#include "omv_common.h"
#include "omv_spi.h"

#define VOSPI_HEADER_WORDS          (2) // 16-bits
#define VOSPI_PID_SIZE_PIXELS       (80) // w, 16-bits per pixel
#define VOSPI_PIDS_PER_SID          (60) // h
#define VOSPI_SIDS_PER_FRAME        (4)
#define VOSPI_PACKET_SIZE           (VOSPI_HEADER_WORDS + VOSPI_PID_SIZE_PIXELS) // 16-bits
#define VOSPI_SID_SIZE_PIXELS       (VOSPI_PIDS_PER_SID * VOSPI_PID_SIZE_PIXELS) // 16-bits

#define VOSPI_BUFFER_SIZE           (VOSPI_PACKET_SIZE * 2) // 16-bits
#define VOSPI_CLOCK_SPEED           20000000 // hz
#define VOSPI_SYNC_MS               200 // ms

#define VOSPI_SPECIAL_PACKET        (20)
#define VOSPI_DONT_CARE_PACKET      (0x0F00)
#define VOSPI_HEADER_DONT_CARE(x)   (((x) & VOSPI_DONT_CARE_PACKET) == VOSPI_DONT_CARE_PACKET)
#define VOSPI_HEADER_PID(id)        ((id) & 0x0FFF)
#define VOSPI_HEADER_SID(id)        (((id) >> 12) & 0x7)

typedef enum {
    VOSPI_FLAGS_CAPTURE = (1 << 0),
    VOSPI_FLAGS_RESYNC  = (1 << 1),
} vospi_flags_t;

typedef struct _vospi_state {
    int pid;
    int sid;
    framebuffer_t *fb;
    bool lepton_3;
    omv_spi_t spi_bus;
    volatile uint32_t flags;
} vospi_state_t;

static vospi_state_t vospi;

static uint16_t OMV_ATTR_SEC_ALIGN(vospi_buf[VOSPI_BUFFER_SIZE], OMV_VOSPI_DMA_BUFFER, OMV_DMA_ALIGNMENT);
static void vospi_callback(omv_spi_t *spi, void *userdata, void *buf);

static void vospi_resync() {
    omv_spi_transfer_t spi_xfer = {
        .rxbuf = vospi_buf,
        .size = VOSPI_BUFFER_SIZE,
        .flags = OMV_SPI_XFER_DMA,
        .callback = vospi_callback,
    };

    mp_hal_delay_ms(VOSPI_SYNC_MS);
    omv_spi_transfer_start(&vospi.spi_bus, &spi_xfer);
}

#if defined(OMV_ENABLE_VOSPI_CRC)
static bool vospi_check_crc(const uint16_t *base) {
    int id = base[0];
    int packet_crc = base[1];
    int crc = ByteCRC16((id >> 8) & 0x0F, 0);
    crc = ByteCRC16(id, crc);
    crc = ByteCRC16(0, crc);
    crc = ByteCRC16(0, crc);

    for (int i = VOSPI_HEADER_WORDS; i < VOSPI_PACKET_SIZE; i++) {
        int value = base[i];
        crc = ByteCRC16(value >> 8, crc);
        crc = ByteCRC16(value, crc);
    }

    return packet_crc == crc;
}
#endif

void vospi_callback(omv_spi_t *spi, void *userdata, void *buf) {
    if (!(vospi.flags & VOSPI_FLAGS_CAPTURE)) {
        return;
    }

    const uint16_t *base = (uint16_t *) buf;

    int id = base[0];

    // Ignore don't care packets.
    if (VOSPI_HEADER_DONT_CARE(id)) {
        return;
    }

    int pid = VOSPI_HEADER_PID(id);
    int sid = VOSPI_HEADER_SID(id) - 1;

    // Discard packets with a pid != 0 when waiting for the first packet.
    if ((vospi.pid == 0) && (pid != 0)) {
        return;
    }

    // Discard sidments with a sid != 0 when waiting for the first segment.
    if (vospi.lepton_3 && (pid == VOSPI_SPECIAL_PACKET) && (vospi.sid == 0) && (sid != 0)) {
        vospi.pid = 0;
        return;
    }

    // Are we in sync with the flir lepton?
    if ((pid != vospi.pid)
    #if defined(OMV_ENABLE_VOSPI_CRC)
        || (!vospi_check_crc(base))
    #endif
        || (vospi.lepton_3 && (pid == VOSPI_SPECIAL_PACKET) && (sid != vospi.sid))) {
        vospi_abort();
        vospi.flags |= VOSPI_FLAGS_CAPTURE;
        return;
    }

    vbuffer_t *buffer = framebuffer_get_tail(vospi.fb, FB_PEEK);

    if (buffer) {
        memcpy(((uint16_t *) buffer->data)
               + (vospi.pid * VOSPI_PID_SIZE_PIXELS)
               + (vospi.sid * VOSPI_SID_SIZE_PIXELS),
               base + VOSPI_HEADER_WORDS, VOSPI_PID_SIZE_PIXELS * sizeof(uint16_t));

        vospi.pid += 1;
        if (vospi.pid == VOSPI_PIDS_PER_SID) {
            vospi.pid = 0;

            // For the FLIR Lepton 3 we have to receive all the pids in all the segments.
            if (vospi.lepton_3) {
                vospi.sid += 1;
                if (vospi.sid == VOSPI_SIDS_PER_FRAME) {
                    vospi.sid = 0;
                    framebuffer_get_tail(vospi.fb, FB_NO_FLAGS);
                }
                // For the FLIR Lepton 1/2 we just have to receive all the pids.
            } else {
                framebuffer_get_tail(vospi.fb, FB_NO_FLAGS);
            }
        }
    } else {
        vospi.flags &= ~VOSPI_FLAGS_CAPTURE;
    }
}

int vospi_init(uint32_t n_packets, framebuffer_t *fb) {
    memset(&vospi, 0, sizeof(vospi_state_t));
    vospi.lepton_3 = n_packets > VOSPI_PIDS_PER_SID;
    vospi.fb = fb;
    // resync on first snapshot.
    vospi.flags = VOSPI_FLAGS_RESYNC;

    omv_spi_config_t spi_config;
    omv_spi_default_config(&spi_config, OMV_CSI_SPI_ID);

    spi_config.bus_mode = OMV_SPI_BUS_RX;
    spi_config.datasize = 16;
    spi_config.baudrate = VOSPI_CLOCK_SPEED;
    spi_config.clk_pol = OMV_SPI_CPOL_HIGH;
    spi_config.clk_pha = OMV_SPI_CPHA_2EDGE;
    spi_config.dma_flags = OMV_SPI_DMA_CIRCULAR | OMV_SPI_DMA_DOUBLE;

    return omv_spi_init(&vospi.spi_bus, &spi_config);
}

int vospi_deinit() {
    return omv_spi_deinit(&vospi.spi_bus);
}

int vospi_abort(void) {
    vospi.flags &= ~VOSPI_FLAGS_CAPTURE;
    int ret = omv_spi_transfer_abort(&vospi.spi_bus);
    vospi.pid = 0;
    vospi.sid = 0;
    vospi.flags |= VOSPI_FLAGS_RESYNC;
    return ret;
}

bool vospi_active(void) {
    return vospi.flags & VOSPI_FLAGS_CAPTURE;
}

int vospi_snapshot(uint32_t timeout_ms) {
    framebuffer_free_current_buffer(vospi.fb);

    if (!(vospi.flags & VOSPI_FLAGS_CAPTURE)) {
        framebuffer_setup_buffers(vospi.fb);

        // Restart counters to capture a new frame.
        vospi.flags |= VOSPI_FLAGS_CAPTURE;
    }

    // Snapshot start tick
    mp_uint_t tick_start = mp_hal_ticks_ms();

    do {
        if (vospi.flags & VOSPI_FLAGS_RESYNC) {
            vospi.flags &= ~VOSPI_FLAGS_RESYNC;
            vospi_resync();
        }

        if (timeout_ms == 0) {
            return -1;
        }

        if ((mp_hal_ticks_ms() - tick_start) > timeout_ms) {
            vospi_abort();
            return -1;
        }

        MICROPY_EVENT_POLL_HOOK
    } while (!framebuffer_get_head(vospi.fb, FB_NO_FLAGS));

    return 0;
}
#endif
