/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "test_manager.h"

int main(int argc, char **argv) {
    try {
        TestManager test_manager;
        if (test_manager.fill_test_param(argc, argv)) {
            test_manager.run_tests();
        }
    } catch (std::exception& e) {
        std::cout << "Exception : " << e.what() << std::endl;
    } catch (const std::ifstream::failure& e) {
        std::cout << "Exception: " << e.what() << std::endl;
    } catch (enn_test::EnnTestReturn err) {
        enn_test::print_result(err);
    }
}
