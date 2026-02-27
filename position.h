#include "consts.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#pragma once

struct Piece {
  U8 data;
  Piece() : data(PieceType::None) {}
  Piece(bool color, PieceType type) {
    if (type == PieceType::None) {
      data = PieceType::None;
    } else {
      data = type | (color << 3);
    }
  }

  inline bool empty() const { return data == 0; }
  inline bool color() const { return (data >> 3) & 1; }
  inline PieceType type() const { return static_cast<PieceType>(data & 7); }
  bool operator==(const Piece &other) const { return data == other.data; }
  bool operator!=(const Piece &other) const { return data != other.data; }
};

struct Move {
  U16 data;
  Move() : data(0) {}
  Move(int from, int to, bool promo, PieceType captured) {
    data = from | (to << 6) | (promo << 12) | (captured << 13);
  }

  inline int from() const { return data & 63; }
  inline int to() const { return (data >> 6) & 63; }
  inline bool promoted() const { return (data >> 12) & 1; }
  inline PieceType captured() const {
    return static_cast<PieceType>((data >> 13) & 7);
  }
  bool operator==(const Move &other) const { return data == other.data; }
  bool operator!=(const Move &other) const { return data != other.data; }
};

struct Position {
  U64 Bitboards[8];
  Piece pieces[64];
  bool stm = 0;
  int halfmovecount = 0;
  U64 zobristhash = 0ULL;
  U64 scratchzobrist();
  void initialize();
  bool twokings();
  bool bareking(int color);
  int material();
  U64 checkers(int color);
  void makemove(Move mov);
  void unmakemove(Move mov);
  int generatemoves(Move *movelist);
  U64 perft(int depth, int initialdepth);
  U64 perftnobulk(int depth, int initialdepth);
  void parseFEN(std::string FEN);
  std::string getFEN();
  bool see_exceeds(Move mov, int threshold);
};

void initializeleaperattacks();
void initializemasks();
void initializerankattacks();
void initializezobrist();
std::string algebraic(int notation);
std::string get129600FEN(int seed1, int seed2);