#include <iostream>
#include <vector>
#include <sys/syscall.h>
#include <linux/perf_event.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <cstring>
#include <iomanip>

class IPCMeasurer
{
    // 基础计数器
    int fd_cycles = -1;
    int fd_instr = -1;
    
    // L1 Data Cache
    int fd_l1d_ref = -1;
    int fd_l1d_miss = -1; // 对应 L1-dcache-load-misses
    
    // L2 Data Cache
    int fd_l2d_ref = -1;
    int fd_l2d_miss = -1;

    // L3 (LLC) Cache
    int fd_l3_ref = -1;
    int fd_l3_miss = -1;

    // dTLB 计数器 (新增)
    int fd_dtlb_ref = -1;  // 对应 dTLB-loads
    int fd_dtlb_miss = -1; // 对应 dTLB-load-misses

    static int perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu, int group_fd, unsigned long flags)
    {
        return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
    }

    // 通用打开计数器函数
    int open_counter(uint32_t type, uint64_t config)
    {
        struct perf_event_attr pe;
        memset(&pe, 0, sizeof(struct perf_event_attr));
        pe.type = type;
        pe.size = sizeof(struct perf_event_attr);
        pe.config = config;
        pe.disabled = 1;
        pe.exclude_kernel = 1;
        pe.exclude_hv = 1;

        int fd = perf_event_open(&pe, 0, -1, -1, 0);
        if (fd == -1)
        {
            perror("Error opening counter");
        }
        return fd;
    }

    // 辅助函数：快速构建 HW_CACHE 的 config 参数
    uint64_t make_hw_cache_config(uint64_t id, uint64_t op, uint64_t result) {
        return id | (op << 8) | (result << 16);
    }

  public:
    void start()
    {
        // 1. 基础硬件计数器
        fd_cycles = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES);
        fd_instr  = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS);

        // 2. L1 Data Cache 计数器 (使用 PERF_TYPE_HW_CACHE)
        fd_l1d_ref  = open_counter(PERF_TYPE_HW_CACHE, make_hw_cache_config(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS));
        fd_l1d_miss = open_counter(PERF_TYPE_HW_CACHE, make_hw_cache_config(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS));

        // 3. L2 Data Cache 计数器
        fd_l2d_ref  = open_counter(PERF_TYPE_HW_CACHE, make_hw_cache_config(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS)); 
        fd_l2d_miss = open_counter(PERF_TYPE_HW_CACHE, make_hw_cache_config(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS));

        // 4. L3 (LLC) 计数器
        fd_l3_ref  = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES);
        fd_l3_miss = open_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES);

        // 5. dTLB 计数器 (新增)
        fd_dtlb_ref  = open_counter(PERF_TYPE_HW_CACHE, make_hw_cache_config(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS));
        fd_dtlb_miss = open_counter(PERF_TYPE_HW_CACHE, make_hw_cache_config(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS));

        const int fds[] = {fd_cycles, fd_instr, fd_l1d_ref, fd_l1d_miss, fd_l2d_ref, fd_l2d_miss, fd_l3_ref, fd_l3_miss, fd_dtlb_ref, fd_dtlb_miss};
        for (int fd : fds)
        {
            if (fd != -1)
            {
                ioctl(fd, PERF_EVENT_IOC_RESET, 0);
                ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
            }
        }
    }

    void stop_and_print()
    {
        const int fds[] = {fd_cycles, fd_instr, fd_l1d_ref, fd_l1d_miss, fd_l2d_ref, fd_l2d_miss, fd_l3_ref, fd_l3_miss, fd_dtlb_ref, fd_dtlb_miss};
        for (int fd : fds)
            if (fd != -1)
                ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);

        long long cycles = 0, instr = 0;
        long long l1d_ref = 0, l1d_miss = 0;
        long long l2d_ref = 0, l2d_miss = 0;
        long long l3_ref = 0, l3_miss = 0;
        long long dtlb_ref = 0, dtlb_miss = 0; // 新增

        // 显式接收返回值以消除 -Werror=unused-result
        auto r1 = read(fd_cycles, &cycles, sizeof(long long));
        auto r2 = read(fd_instr, &instr, sizeof(long long));
        auto r3 = read(fd_l1d_ref, &l1d_ref, sizeof(long long));
        auto r4 = read(fd_l1d_miss, &l1d_miss, sizeof(long long));
        auto r5 = read(fd_l2d_ref, &l2d_ref, sizeof(long long));
        auto r6 = read(fd_l2d_miss, &l2d_miss, sizeof(long long));
        auto r7 = read(fd_l3_ref, &l3_ref, sizeof(long long));
        auto r8 = read(fd_l3_miss, &l3_miss, sizeof(long long));
        auto r9 = read(fd_dtlb_ref, &dtlb_ref, sizeof(long long));   // 新增
        auto r10 = read(fd_dtlb_miss, &dtlb_miss, sizeof(long long)); // 新增

        if (r1 > 0 && r2 > 0 && cycles > 0)
        {
            double ipc = (double)instr / cycles;
            double l1d_miss_rate  = (l1d_ref > 0)  ? (double)l1d_miss / l1d_ref * 100.0 : 0.0;
            double l2d_miss_rate  = (l2d_ref > 0)  ? (double)l2d_miss / l2d_ref * 100.0 : 0.0;
            double l3_miss_rate   = (l3_ref > 0)   ? (double)l3_miss / l3_ref * 100.0 : 0.0;
            double dtlb_miss_rate = (dtlb_ref > 0) ? (double)dtlb_miss / dtlb_ref * 100.0 : 0.0; // 新增

            std::cout << std::fixed << std::setprecision(3);
            std::cout << "\n======= Cache Hierarchy & TLB Analysis =======" << std::endl;
            std::cout << "IPC:              " << ipc << (ipc < 1.0 ? " [Low!]" : "") << std::endl;
            std::cout << "----------------------------------------" << std::endl;
            std::cout << "L1D Cache Misses: " << l1d_miss << " / " << l1d_ref << " (" << l1d_miss_rate << "%) [L1-dcache-load-misses]" << std::endl;
            std::cout << "L2D Cache Misses: " << l2d_miss << " / " << l2d_ref << " (" << l2d_miss_rate << "%)" << std::endl;
            std::cout << "L3  Cache Misses: " << l3_miss  << " / " << l3_ref  << " (" << l3_miss_rate  << "%)" << std::endl;
            std::cout << "dTLB Load Misses: " << dtlb_miss << " / " << dtlb_ref << " (" << dtlb_miss_rate << "%) [dTLB-loads / misses]" << std::endl;
            std::cout << "========================================\n" << std::endl;
        }

        for (int fd : fds)
            if (fd != -1)
                close(fd);
    }
};