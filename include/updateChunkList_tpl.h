#include "templates.h"
#include <stdlib.h>

#define CHUNKLENGTH 1024
typedef struct TEMPLATE(_spray_updchunk_struct,T) {
  int indices[CHUNKLENGTH];
  T content[CHUNKLENGTH];
  int top;
  struct TEMPLATE(_spray_updchunk_struct,T)* next;
} TEMPLATE(_spray_updchunk,T);

typedef struct {
  TEMPLATE(_spray_updchunk,T)* firstchunk;
  TEMPLATE(_spray_updchunk,T)* lastchunk;
} TEMPLATE(_spray_updlist,T);

TEMPLATE(_spray_updchunk,T)* TEMPLATE(_spray_updchunk_create,T)() {
  TEMPLATE(_spray_updchunk,T)* chunk = (TEMPLATE(_spray_updchunk,T)*)malloc(sizeof(TEMPLATE(_spray_updchunk,T)));
  chunk->top = 0;
  chunk->next = NULL;
  return chunk;
}

void TEMPLATE(_spray_updlist_init,T)(TEMPLATE(_spray_updlist,T)* list) {
  TEMPLATE(_spray_updchunk,T)* chunk = TEMPLATE(_spray_updchunk_create,T)();
  list->firstchunk = chunk;
  list->lastchunk = chunk;
}

void TEMPLATE(_spray_updlist_append,T)(TEMPLATE(_spray_updlist,T)* list, int idx, T val) {
  TEMPLATE(_spray_updchunk,T)* lastchunk = list->lastchunk;
  if(lastchunk->top == CHUNKLENGTH) {
    TEMPLATE(_spray_updchunk,T)* newchunk = TEMPLATE(_spray_updchunk_create,T)();
    lastchunk->next = newchunk;
    lastchunk = newchunk;
    list->lastchunk = newchunk;
  }
  lastchunk->content[lastchunk->top] = val;
  lastchunk->indices[lastchunk->top] = idx;
  lastchunk->top ++;
}

void TEMPLATE(_spray_updlist_commit,T)(TEMPLATE(_spray_updlist,T)* list, T* target) {
  TEMPLATE(_spray_updchunk,T)* chunk = list->firstchunk;
  while(chunk) {
    for(int i=0; i<chunk->top; i++) {
      target[chunk->indices[i]] += chunk->content[i];
    }
    TEMPLATE(_spray_updchunk,T)* oldchunk = chunk;
    chunk = chunk->next;
    free(oldchunk);
  }
}

