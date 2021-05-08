#ifndef DYNAMICKEEPERREDUCTION_H
#define DYNAMICKEEPERREDUCTION_H


typedef struct {
  int* owner_start;
  int* owner;
} ownership_sequence;

  #ifdef T
  #undef T
  #endif
  #define T float
  #include "dynamicKeeperReduction_tpl.h"
  
  #ifdef T
  #undef T
  #endif
  #define T double
  #include "dynamicKeeperReduction_tpl.h"

#endif

