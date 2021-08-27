#!/bin/bash

# product info
# n_[name] = ( [name] [build name] [lunch name] [build platform ver.] [HIDL ver] )
n_universal2100=('universal2100' 'universal2100' 'full_universal2100_r-eng' 'R' '1.0')
n_universal2100s=('universal2100s' 'universal2100s' 'full_universal2100_s-eng' 'S' '1.0')
n_erd9925=('erd9925' 'erd9925' 'full_erd9925_s-eng' 'S' '1.0')
n_universal9925=('universal9925' 'universal9925' 'full_universal9925_s-eng' 'S' '1.0')
n_target_lists=('n_universal2100' 'n_universal2100s' 'n_erd9925' 'n_universal9925')

# API for getting product information
function init_product_info()
{
  bd_name="n_$1"
  declare -n bd_info="$bd_name"
  name="${bd_info[0]}"

  if [[ $1 != ${name} ]]; then
    echo " # ERROR: There's no name in DB ($1 != ${name})"
    verified=false
    return
  fi

  echo ""
  echo "=================================================="
  echo " # Build configuration "
  echo "=================================================="
  echo " # MC_PRODUCT_NAME=${bd_info[0]}"
  echo " # MC_BUILD_NAME=${bd_info[1]}"
  echo " # MC_LUNCH_NAME=${bd_info[2]}"
  echo " # MC_BUILD_PLATFORM_VER=${bd_info[3]}"
  echo " # MC_HIDL_VER=${bd_info[4]}"
  echo "=================================================="
  echo ""

  MC_PRODUCT_NAME=${bd_info[0]}
  MC_BUILD_NAME=${bd_info[1]}
  MC_LUNCH_NAME=${bd_info[2]}
  MC_BUILD_PLATFORM_VER=${bd_info[3]}
  MC_HIDL_VER=${bd_info[4]}

  verified=true
  return
}

