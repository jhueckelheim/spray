#ifndef NDBLOCKREDUCTION_H
#define NDBLOCKREDUCTION_H

  #ifdef T
  #undef T
  #endif
  #define T float
  #include "ndblockReduction_tpl.h"
  
  #ifdef T
  #undef T
  #endif
  #define T double
  #include "ndblockReduction_tpl.h"

#endif
