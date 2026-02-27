#include "consts.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#pragma once

struct Position {
  U64 Bitboards[8];
  int pieces[64];
  bool stm = 0;
  int halfmovecount = 0;
  U64 zobristhash = 0ULL;
  U64 scratchzobrist();
  void initialize();
  bool twokings();
  bool bareking(int color);
  int material();
  U64 checkers(int color);
  U64 keyafter(int notation);
  void makemove(int notation);
  void unmakemove(int notation);
  int generatemoves(int color, bool capturesonly, int *movelist);
  U64 perft(int depth, int initialdepth, int color);
  U64 perftnobulk(int depth, int initialdepth, int color);
  void parseFEN(std::string FEN);
  std::string getFEN();
  bool see_exceeds(int mov, int threshold);
};

void initializeleaperattacks();
void initializemasks();
void initializerankattacks();
void initializezobrist();
std::string algebraic(int notation);
std::string get129600FEN(int seed1, int seed2);