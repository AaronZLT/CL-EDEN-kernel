/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

#include "userdriver/cpu/cpu_op_executor.h"
#include "userdriver/common/operator_interfaces/OperatorInterfaces.h"

namespace enn {
namespace ud {

DEFINE_EXECUTOR(IAsymmDequantization, BUF_IN(0), BUF_OUT(0))
DEFINE_EXECUTOR(IAsymmQuantization, BUF_IN(0), BUF_OUT(0))
DEFINE_EXECUTOR(ICFUConverter, BUF_IN(0), BUF_OUT(0))
DEFINE_EXECUTOR(ICFUInverter, BUF_IN(0), BUF_OUT(0))
DEFINE_EXECUTOR(IConcat, {BUF_IN(0), BUF_IN(1)}, BUF_OUT(0))
DEFINE_EXECUTOR(IDequantization, BUF_IN(0), BUF_OUT(0))
#if 0  // Required, ToDo(empire.jung, 8/31): Legacy SSD, remove after deciding to use priorbox
DEFINE_EXECUTOR(IDetection, BUF_IN(1), BUF_IN(2), BUF_IN(0), BUF_OUT(0))
#else
DEFINE_EXECUTOR(IDetection, BUF_IN(0), BUF_IN(1), ARG_DATA(0), BUF_OUT(0))
#endif
DEFINE_EXECUTOR(IFlatten, BUF_IN(0), BUF_OUT(0))
DEFINE_EXECUTOR(INormalization, BUF_IN(0), BUF_OUT(0))
DEFINE_EXECUTOR(INormalQuantization, BUF_IN(0), BUF_OUT(0))
DEFINE_EXECUTOR(IPad, BUF_IN(0), BUF_OUT(0))
DEFINE_EXECUTOR(IQuantization, BUF_IN(0), BUF_OUT(0))
DEFINE_EXECUTOR(ISigDet, BUF_IN(1), BUF_IN(2), BUF_IN(0), BUF_OUT(0))
DEFINE_EXECUTOR(ISoftmax, BUF_IN(0), BUF_OUT(0))

}  // namespace ud
}  // namespace enn
