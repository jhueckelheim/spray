#ifndef UPDATECHUNK_H
#define UPDATECHUNK_H

  #ifdef T
  #undef T
  #endif
  #define T float
  #include "updateChunkList_tpl.h"
  
  #ifdef T
  #undef T
  #endif
  #define T double
  #include "updateChunkList_tpl.h"

#endif

