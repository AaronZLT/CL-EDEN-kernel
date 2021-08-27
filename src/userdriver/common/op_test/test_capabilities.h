#pragma once

namespace enn {
namespace platform {

// Nice to have, ToDo(empire.jung, TBD): Optimize chipset list. If not needed, it will be removed.
constexpr const char* EXYNOS9925 = "s5e9925";       // Pamir
constexpr const char* EXYNOS991 = "exynos991";      // Olympus
constexpr const char* EXYNOS2100 = "exynos2100";    // Olympus
constexpr const char* EXYNOS9820 = "exynos9820";    // Makalu
constexpr const char* EXYNOS9825 = "exynos9825";    // Makalu
constexpr const char* EXYNOS9830 = "exynos9830";    // 2020
constexpr const char* EXYNOS990 = "exynos990";      // 2020
constexpr const char* EXYNOS9630 = "exynos9630";    // Neus
constexpr const char* EXYNOS980 = "exynos980";      // Neus
constexpr const char* EXYNOS880 = "exynos880";      // Neus
constexpr const char* EXYNOS3830 = "exynos3830";    // Nacho
constexpr const char* HR80 = "hr80";                // KITT
constexpr const char* EXYNOSAUTO9 = "exynosauto9";  // KITT Android
constexpr const char* EXYNOS9840 = "exynos9840";    // Olympus
constexpr const char* EXYNOS9815 = "s5e9815";       // Orange

#ifdef VELOCE_SOC
constexpr const char* VELOCE_EXYNOS9925 = "ranchu_veloce_exynos9925";  // Veloce Pamir
constexpr const char* VELOCE_EXYNOS991 = "ranchu_veloce_exynos991";    // Veloce Olympus
#endif

}  // namespace platform
}  // namespace enn