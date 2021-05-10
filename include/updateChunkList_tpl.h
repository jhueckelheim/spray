#include "templates.h"
#include <stdlib.h>

#define CHUNKLENGTH 1023
typedef struct TEMPLATE(_spray_updchunk_struct,T) {
  struct {
    int index;
    T content;
  } entries [CHUNKLENGTH];
  struct TEMPLATE(_spray_updchunk_struct,T)* next;
} TEMPLATE(_spray_updchunk,T);

typedef struct {
  TEMPLATE(_spray_updchunk,T)* firstchunk;
  TEMPLATE(_spray_updchunk,T)* lastchunk;
  int top;
} TEMPLATE(_spray_updlist,T);

TEMPLATE(_spray_updchunk,T)* TEMPLATE(_spray_updchunk_create,T)() {
  TEMPLATE(_spray_updchunk,T)* chunk = (TEMPLATE(_spray_updchunk,T)*)malloc(sizeof(TEMPLATE(_spray_updchunk,T)));
  chunk->next = NULL;
  return chunk;
}

void TEMPLATE(_spray_updlist_init,T)(TEMPLATE(_spray_updlist,T)* list) {
  TEMPLATE(_spray_updchunk,T)* chunk = TEMPLATE(_spray_updchunk_create,T)();
  list->firstchunk = chunk;
  list->lastchunk = chunk;
  list->top = 0;
}

void TEMPLATE(_spray_updlist_append,T)(TEMPLATE(_spray_updlist,T)* list, int idx, T val) {
  TEMPLATE(_spray_updchunk,T)* lastchunk = list->lastchunk;
  if(list->top == CHUNKLENGTH) {
    TEMPLATE(_spray_updchunk,T)* newchunk = TEMPLATE(_spray_updchunk_create,T)();
    lastchunk->next = newchunk;
    lastchunk = newchunk;
    list->lastchunk = newchunk;
    list->top = 0;
  }
  lastchunk->entries[list->top].index = idx;
  lastchunk->entries[list->top].content = val;
  list->top++;
}

void TEMPLATE(_spray_updlist_commit,T)(TEMPLATE(_spray_updlist,T)* list, T* target) {
  TEMPLATE(_spray_updchunk,T)* chunk = list->firstchunk;
  while(chunk) {
    TEMPLATE(_spray_updchunk,T)* nextchunk = chunk->next;
    int e = nextchunk ? CHUNKLENGTH : list->top;
    for(int i=0; i< e; i++) {
      target[chunk->entries[i].index] += chunk->entries[i].content;
    }
    free(chunk);
    chunk = nextchunk;
  }
}

