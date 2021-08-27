#yaml
dsp_debug_scenario:
    name: IV3_cmdq_test
    local:
      model_file: dsp_iv3/iv3_dirty.nnc
      input:
        - dsp_iv3/nnc_input_0.bin
      golden:
        - dsp_iv3/nnc_output_0.bin
    target:
      model_file: /data/raw/iv3_dirty.nnc
      input:
        - /data/raw/nnc_input_0.bin
      golden:
        - /data/raw/nnc_output_0.bin
    max_layer: 108
