#ifndef DYNAMICKEEPERREDUCTION_H
#define DYNAMICKEEPERREDUCTION_H

typedef char * Bitarray;

Bitarray bitarray_create(int Size) {
  return (Bitarray)calloc((Size + 7) / 8, 1);
}
void bitarray_set(Bitarray BA, int Idx) {
  int CharIdx = Idx / 8;
  int ByteIdx = Idx % 8;
  BA[CharIdx] |= 1 << ByteIdx;
}
int bitarray_get(Bitarray BA, int Idx) {
  int CharIdx = Idx / 8;
  int ByteIdx = Idx % 8;
  return BA[CharIdx] & (1 << ByteIdx);
}


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

