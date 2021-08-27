---
Title: README.md / medium
Output: pdf_document
author: Hoon Choi
Date: 2020. 12. 30
Reference: http://pdf.plantuml.net/PlantUML_Language_Reference_Guide_ko.pdf
---
<i>This document simply describe API use cases</i>

# Description of medium in Exynos NN framework

- Android framework should have daemon service which has an access permission to a vendor device.
- For various usage, we prepare a medium between a client and service core as several types decided in build time

```plantUML
@startuml
start
:UserAPI;
:ContextManager;
:(parser), ..., (optimizer);
:Medium Interface (client);
partition "MEDIUM "{
fork
  ->Android;
  :Medium interface convert;
  :General Interface;
  :Medium interface deconvert;
  ->android;
fork again
  ->Linux;
  :direct call;
end fork
}
:Medium Interface (service core);
:(service core);
stop
@enduml
```