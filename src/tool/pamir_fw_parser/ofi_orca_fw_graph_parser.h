
/*
 * Copyright (C) 2019 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 *
 */

#ifndef SOURCE_TOOLS_ORCA_FW_PARSER_OFI_ORCA_FW_GRAPH_PARSER_H_
#define SOURCE_TOOLS_ORCA_FW_PARSER_OFI_ORCA_FW_GRAPH_PARSER_H_
#include <vector>
#ifdef OFI
#include "dal_rt.h"

using namespace dal;
#endif
extern int get_size_of_jib_pool(void *dsp_graph);

extern int parse_dsp_fbs(char *cp_jib_arr, void *dsp_graph,
                         const std::vector<struct KernelBinInfo> &bin_info,
                         uint32_t graph_id, uint32_t dsp_graph_size = 0);

extern void show_jib_pool(char *cp_jib_arr);

#endif  // SOURCE_TOOLS_ORCA_FW_PARSER_OFI_ORCA_FW_GRAPH_PARSER_H_


/*
int main() {
  int jib_pool_size;
  get_size_of_jib_pool(&jib_pool_size);
  std::cout << "jib_pool_size: " << jib_pool_size << std::endl;
  std::cout << "..." << std::endl;
  uint8_t *pool = new uint8_t[jib_pool_size];
  void *fbs_root;
  parse_dsp_fbs(pool, fbs_root, 0);
  delete [] pool;

  return 0;
}
*/
