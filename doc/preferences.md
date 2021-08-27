---
Title: Preferneces
Output: pdf_document
author: Hoon Choi
Date: 2021. 07. 02
Reference: https://ogom.github.io/draw_uml/plantuml/
---

#### Preferences with Open?
> User can set preference with OpenModel(model, preferences)
> For find-grained control, user can also call OpenModelExtension(model, extended preferences)

#### Preferences
> The framework manages preference as follow:

* enn_api-type.h
```c++
typedef enum {
  NORMAL_MODE,
  BOOST_MODE,
  BOOST_ON_EXECUTE,
} PerfModePreference;

typedef struct ennModelPreference {
    uint32_t preset_id;  // default
    PerfModePreference mode;
    uint32_t custom[8];
} EnnModelPreference;
```

* enn_common_type.h (internal)
```c++
using openModelPreference = ennModelExtendedPreference;
```

* enn_api.h
```c++
typedef struct ennModelExtendedPreference {
    uint32_t preset_id = 0;  // default
    PerfModePreference mode = BOOST_ON_EXECUTE;
    uint32_t target_latency = 0;
    uint32_t tile_num = 1;
    uint32_t core_affinity = 0xFFFFFFFF;
    uint32_t priority = 0;
    uint32_t custom[8];
} EnnModelExtendedPreference;

extern EnnModelId EnnOpenModelExtension(const char* model_file, const 
EnnModelExtendedPreference& preference);

```