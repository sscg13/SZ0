#include "position.h"
#include "consts.h"
U64 KingAttacks[64];
U64 PawnAttacks[2][64];
U64 AlfilAttacks[64];
U64 FerzAttacks[64];
U64 KnightAttacks[64];
U64 RankMask[64];
U64 FileMask[64];
U64 RankAttacks[512];
U64 hashes[8][64];
const U64 colorhash = 0xE344F58E0F3B26E5;
U64 shift_w(U64 bitboard) { return (bitboard & ~FileA) >> 1; }
U64 shift_n(U64 bitboard) { return bitboard << 8; }
U64 shift_s(U64 bitboard) { return bitboard >> 8; }
U64 shift_e(U64 bitboard) { return (bitboard & ~FileH) << 1; }
void initializeleaperattacks() {
  for (int i = 0; i < 64; i++) {
    U64 square = 1ULL << i;
    PawnAttacks[0][i] = ((square & ~FileA) << 7) | ((square & ~FileH) << 9);
    PawnAttacks[1][i] = ((square & ~FileA) >> 9) | ((square & ~FileH) >> 7);
    U64 horizontal = square | shift_w(square) | shift_e(square);
    U64 full = horizontal | shift_n(horizontal) | shift_s(horizontal);
    KingAttacks[i] = full & ~square;
    U64 knightattack = ((square & ~FileA) << 15);
    knightattack |= ((square & ~FileA) >> 17);
    knightattack |= ((square & ~FileH) << 17);
    knightattack |= ((square & ~FileH) >> 15);
    knightattack |= ((square & ~FileG & ~FileH) << 10);
    knightattack |= ((square & ~FileG & ~FileH) >> 6);
    knightattack |= ((square & ~FileA & ~FileB) << 6);
    knightattack |= ((square & ~FileA & ~FileB) >> 10);
    KnightAttacks[i] = knightattack;
    FerzAttacks[i] = shift_n(shift_w(square) | shift_e(square)) |
                     shift_s(shift_w(square) | shift_e(square));
    U64 alfilattack = ((square & ~FileA & ~FileB) << 14);
    alfilattack |= ((square & ~FileA & ~FileB) >> 18);
    alfilattack |= ((square & ~FileG & ~FileH) << 18);
    alfilattack |= ((square & ~FileG & ~FileH) >> 14);
    AlfilAttacks[i] = alfilattack;
  }
}
void initializemasks() {
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      RankMask[8 * i + j] = Ranks[i];
      FileMask[8 * i + j] = Files[j];
    }
  }
}
void initializerankattacks() {
  for (U64 i = 0ULL; i < 64; i++) {
    U64 occupied = i << 1;
    for (int j = 0; j < 8; j++) {
      U64 attacks = 0ULL;
      if (j > 0) {
        int k = j - 1;
        while (k >= 0) {
          attacks |= (1ULL << k);
          if ((1ULL << k) & occupied) {
            k = 0;
          }
          k--;
        }
      }
      if (j < 7) {
        int k = j + 1;
        while (k <= 7) {
          attacks |= (1ULL << k);
          if ((1ULL << k) & occupied) {
            k = 7;
          }
          k++;
        }
      }
      RankAttacks[8 * i + j] = attacks;
    }
  }
}
U64 FileAttacks(U64 occupied, int square) {
  U64 forwards = occupied & FileMask[square];
  U64 backwards = __builtin_bswap64(forwards);
  forwards = forwards - 2 * (1ULL << square);
  backwards = backwards - 2 * (1ULL << (56 ^ square));
  backwards = __builtin_bswap64(backwards);
  return (forwards ^ backwards) & FileMask[square];
}
U64 GetRankAttacks(U64 occupied, int square) {
  int row = square & 56;
  int file = square & 7;
  int relevant = (occupied >> (row + 1)) & 63;
  return (RankAttacks[8 * relevant + file] << row);
}
void initializezobrist() {
  std::mt19937_64 mt(20346892);
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 64; j++) {
      hashes[i][j] = mt();
    }
  }
}
std::string algebraic(int notation) {
  std::string convert[64] = {
      "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2",
      "d2", "e2", "f2", "g2", "h2", "a3", "b3", "c3", "d3", "e3", "f3",
      "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4", "a5",
      "b5", "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6",
      "e6", "f6", "g6", "h6", "a7", "b7", "c7", "d7", "e7", "f7", "g7",
      "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"};
  std::string header = convert[notation & 63] + convert[(notation >> 6) & 63];
  if (notation & (1 << 20)) {
    header = header + "q";
  }
  return header;
}
std::string backrow(int seed, bool black) {
  int triangle1[5] = {-12, 0, 1, 2, 4};
  int orders[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  char rank[9] = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '\0'};
  int location = 2 + 4 * (seed % 2);
  rank[location] = black ? 'b' : 'B';
  orders[location] = 0;
  for (int i = location + 1; i < 8; i++) {
    orders[i] = std::max(orders[i] - 1, 0);
  }
  seed /= 2;
  location = 1 + 4 * (seed % 2);
  rank[location] = black ? 'b' : 'B';
  orders[location] = 0;
  for (int i = location + 1; i < 8; i++) {
    orders[i] = std::max(orders[i] - 1, 0);
  }
  seed /= 2;
  int aux = -1;
  for (int i = 0; i < 8; i += 2) {
    if (orders[i] > 0) {
      aux++;
      if (aux == seed % 3) {
        location = i;
      }
    }
  }
  rank[location] = black ? 'q' : 'Q';
  orders[location] = 0;
  for (int i = location + 1; i < 8; i++) {
    orders[i] = std::max(orders[i] - 1, 0);
  }
  seed /= 3;
  aux = -1;
  for (int i = 0; i < 8; i++) {
    if (orders[i] > 0) {
      aux++;
      if (aux == seed % 5) {
        location = i;
      }
    }
  }
  rank[location] = black ? 'k' : 'K';
  orders[location] = 0;
  for (int i = location + 1; i < 8; i++) {
    orders[i] = std::max(orders[i] - 1, 0);
  }
  seed /= 5;
  for (int i = 0; i < 7; i++) {
    for (int j = i + 1; j < 8; j++) {
      if (triangle1[orders[i]] + triangle1[orders[j]] == seed + 1) {
        rank[i] = black ? 'n' : 'N';
        rank[j] = black ? 'n' : 'N';
        orders[i] = 0;
        orders[j] = 0;
      }
    }
  }
  for (int i = 0; i < 8; i++) {
    if (orders[i] > 0) {
      rank[i] = black ? 'r' : 'R';
    }
  }
  std::string output = rank;
  return output;
}
std::string get129600FEN(int seed1, int seed2) {
  std::string FEN = backrow(seed1, 1);
  FEN += "/pppppppp/8/8/8/8/PPPPPPPP/";
  FEN += backrow(seed2, 0);
  FEN += " w - - 0 1";
  return FEN;
}
U64 Position::scratchzobrist() {
  U64 scratch = 0ULL;
  for (int i = 0; i < 64; i++) {
    int piece = pieces[i];
    if (piece > 0) {
      scratch ^= hashes[piece / 8][i];
      scratch ^= hashes[piece % 8][i];
    }
  }
  if (auxinfo & 1) {
    scratch ^= colorhash;
  }
  return scratch;
}
void Position::initialize() {
  Bitboards[0] = Rank1 | Rank2;
  Bitboards[1] = Rank7 | Rank8;
  Bitboards[2] = Rank2 | Rank7;
  Bitboards[3] = (Rank1 | Rank8) & (FileC | FileF);
  Bitboards[4] = (Rank1 | Rank8) & FileE;
  Bitboards[5] = (Rank1 | Rank8) & (FileB | FileG);
  Bitboards[6] = (Rank1 | Rank8) & (FileA | FileH);
  Bitboards[7] = (Rank1 | Rank8) & FileD;
  auxinfo = 0;
  history[0] = auxinfo;
  for (int i = 0; i < 16; i++) {
    pieces[i] = 2 + startpiece[i];
    pieces[56 ^ i] = 10 + startpiece[i];
  }
  for (int i = 16; i < 48; i++) {
    pieces[i] = 0;
  }
  gamephase[0] = 24;
  gamephase[1] = 24;
  gamelength = 0;
  zobrist[0] = zobristhash = scratchzobrist();
}
int Position::repetitions() {
  int repeats = 0;
  for (int i = gamelength - 2; i >= 0; i -= 2) {
    if (zobrist[i] == zobrist[gamelength]) {
      repeats++;
      if (i >= root) {
        repeats++;
      }
    }
  }
  return repeats;
}
int Position::halfmovecount() { return (auxinfo >> 1); }
bool Position::twokings() { return (Bitboards[0] | Bitboards[1]) == Bitboards[7]; }
bool Position::bareking(int color) {
  return (Bitboards[color] & Bitboards[7]) == Bitboards[color];
}
int Position::material() {
  return __builtin_popcountll(Bitboards[2]) +
         __builtin_popcountll(Bitboards[3]) +
         2 * __builtin_popcountll(Bitboards[4]) +
         4 * __builtin_popcountll(Bitboards[5]) +
         6 * __builtin_popcountll(Bitboards[6]);
}
U64 Position::checkers(int color) {
  int kingsquare = __builtin_ctzll(Bitboards[color] & Bitboards[7]);
  int opposite = color ^ 1;
  U64 attacks = 0ULL;
  U64 occupied = Bitboards[0] | Bitboards[1];
  attacks |= (KnightAttacks[kingsquare] & Bitboards[5]);
  attacks |= (PawnAttacks[color][kingsquare] & Bitboards[2]);
  attacks |= (AlfilAttacks[kingsquare] & Bitboards[3]);
  attacks |= (FerzAttacks[kingsquare] & Bitboards[4]);
  attacks |= (GetRankAttacks(occupied, kingsquare) & Bitboards[6]);
  attacks |= (FileAttacks(occupied, kingsquare) & Bitboards[6]);
  attacks &= Bitboards[opposite];
  return attacks;
}
void Position::makenullmove() {
  gamelength++;
  int halfmove = (auxinfo >> 1) & 127;
  zobristhash ^= colorhash;
  auxinfo ^= (halfmove << 1);
  halfmove++;
  auxinfo ^= (halfmove << 1);
  auxinfo ^= 1;
  zobrist[gamelength] = zobristhash;
  history[gamelength] = auxinfo;
}
void Position::unmakenullmove() {
  gamelength--;
  auxinfo = history[gamelength];
  zobristhash = zobrist[gamelength];
}
U64 Position::keyafter(int notation) {
  int from = notation & 63;
  int to = (notation >> 6) & 63;
  int color = (notation >> 12) & 1;
  int piece = (notation >> 13) & 7;
  int captured = (notation >> 17) & 7;
  int promoted = (notation >> 20) & 1;
  int piece2 = (promoted > 0) ? 4 : piece;
  U64 change = (colorhash ^ hashes[color][from] ^ hashes[color][to] ^
                hashes[piece][from] ^ hashes[piece2][to]);
  if (captured) {
    change ^= (hashes[color ^ 1][to] ^ hashes[captured][to]);
  }
  return zobristhash ^ change;
}
void Position::makemove(int notation, bool reversible) {
  // 6 bits from square, 6 bits to square, 1 bit color, 3 bits piece moved, 1
  // bit capture, 3 bits piece captured, 1 bit promotion, 21 bits total

  // 1 bit color, 7 bits halfmove
  gamelength++;
  int from = notation & 63;
  int to = (notation >> 6) & 63;
  int color = (notation >> 12) & 1;
  int piece = (notation >> 13) & 7;
  Bitboards[color] ^= ((1ULL << from) | (1ULL << to));
  Bitboards[piece] ^= ((1ULL << from) | (1ULL << to));
  pieces[to] = pieces[from];
  pieces[from] = 0;
  zobristhash ^= (hashes[color][from] ^ hashes[color][to]);
  zobristhash ^= (hashes[piece][from] ^ hashes[piece][to]);
  int captured = (notation >> 17) & 7;
  int halfmove = (auxinfo >> 1);
  auxinfo ^= (halfmove << 1);
  halfmove++;
  auxinfo &= 0x0003C0FF;
  if (piece == 2) {
    halfmove = 0;
    if (!reversible) {
      gamelength = 0;
    }
  }
  if (notation & (1 << 16)) {
    Bitboards[color ^ 1] ^= (1ULL << to);
    Bitboards[captured] ^= (1ULL << to);
    zobristhash ^= (hashes[color ^ 1][to] ^ hashes[captured][to]);
    halfmove = 0;
    if (!reversible) {
      gamelength = 0;
    }
  }
  if (notation & (1 << 20)) {
    Bitboards[2] ^= (1ULL << to);
    Bitboards[4] ^= (1ULL << to);
    pieces[to] = 8 * color + 4;
    zobristhash ^= (hashes[2][to] ^ hashes[4][to]);
  }
  if (!reversible) {
    root = gamelength;
  }
  auxinfo ^= 1;
  auxinfo ^= (halfmove << 1);
  zobristhash ^= colorhash;
  history[gamelength] = auxinfo;
  zobrist[gamelength] = zobristhash;
  nodecount++;
}
void Position::unmakemove(int notation) {
  gamelength--;
  auxinfo = history[gamelength];
  zobristhash = zobrist[gamelength];
  int from = notation & 63;
  int to = (notation >> 6) & 63;
  int color = (notation >> 12) & 1;
  int piece = (notation >> 13) & 7;
  Bitboards[color] ^= ((1ULL << from) | (1ULL << to));
  Bitboards[piece] ^= ((1ULL << from) | (1ULL << to));
  pieces[from] = pieces[to];
  pieces[to] = 0;
  int captured = (notation >> 17) & 7;
  if (notation & (1 << 16)) {
    Bitboards[color ^ 1] ^= (1ULL << to);
    Bitboards[captured] ^= (1ULL << to);
    pieces[to] = 8 * (color ^ 1) + captured;
  }
  if (notation & (1 << 20)) {
    Bitboards[2] ^= (1ULL << to);
    Bitboards[4] ^= (1ULL << to);
    pieces[from] = 8 * color + 2;
  }
}
int Position::generatemoves(int color, bool capturesonly, int *movelist) {
  int movecount = 0;
  int kingsquare = __builtin_ctzll(Bitboards[color] & Bitboards[7]);
  int pinrank = kingsquare & 56;
  int pinfile = kingsquare & 7;
  int opposite = color ^ 1;
  U64 opponentattacks = 0ULL;
  U64 pinnedpieces = 0ULL;
  U64 checkmask = 0ULL;
  U64 preoccupied = Bitboards[0] | Bitboards[1];
  U64 kingRank = GetRankAttacks(preoccupied, kingsquare);
  U64 kingFile = FileAttacks(preoccupied, kingsquare);
  U64 occupied = preoccupied ^ (1ULL << kingsquare);
  U64 opponentpawns = Bitboards[opposite] & Bitboards[2];
  U64 opponentalfils = Bitboards[opposite] & Bitboards[3];
  U64 opponentferzes = Bitboards[opposite] & Bitboards[4];
  U64 opponentknights = Bitboards[opposite] & Bitboards[5];
  U64 opponentrooks = Bitboards[opposite] & Bitboards[6];
  int pawncount = __builtin_popcountll(opponentpawns);
  int alfilcount = __builtin_popcountll(opponentalfils);
  int ferzcount = __builtin_popcountll(opponentferzes);
  int knightcount = __builtin_popcountll(opponentknights);
  int rookcount = __builtin_popcountll(opponentrooks);
  U64 ourcaptures = 0ULL;
  U64 ourmoves = 0ULL;
  U64 ourmask = 0ULL;
  for (int i = 0; i < pawncount; i++) {
    int pawnsquare = __builtin_ctzll(opponentpawns);
    opponentattacks |= PawnAttacks[opposite][pawnsquare];
    opponentpawns ^= (1ULL << pawnsquare);
  }
  U64 opponentpawnattacks = opponentattacks;
  for (int i = 0; i < alfilcount; i++) {
    int alfilsquare = __builtin_ctzll(opponentalfils);
    opponentattacks |= AlfilAttacks[alfilsquare];
    opponentalfils ^= (1ULL << alfilsquare);
  }
  U64 opponentalfilattacks = opponentattacks;
  for (int i = 0; i < ferzcount; i++) {
    int ferzsquare = __builtin_ctzll(opponentferzes);
    opponentattacks |= FerzAttacks[ferzsquare];
    opponentferzes ^= (1ULL << ferzsquare);
  }
  U64 opponentferzattacks = opponentattacks;
  for (int i = 0; i < knightcount; i++) {
    int knightsquare = __builtin_ctzll(opponentknights);
    opponentattacks |= KnightAttacks[knightsquare];
    opponentknights ^= (1ULL << knightsquare);
  }
  U64 opponentknightattacks = opponentattacks;
  for (int i = 0; i < rookcount; i++) {
    int rooksquare = __builtin_ctzll(opponentrooks);
    U64 r = GetRankAttacks(occupied, rooksquare);
    U64 file = FileAttacks(occupied, rooksquare);
    if (!(r & (1ULL << kingsquare))) {
      pinnedpieces |= (r & kingRank);
    } else {
      checkmask |= (GetRankAttacks(preoccupied, rooksquare) & kingRank);
    }
    if (!(file & (1ULL << kingsquare))) {
      pinnedpieces |= (file & kingFile);
    } else {
      checkmask |= (FileAttacks(preoccupied, rooksquare) & kingFile);
    }
    opponentattacks |= (r | file);
    opponentrooks ^= (1ULL << rooksquare);
  }
  int opponentking = __builtin_ctzll(Bitboards[opposite] & Bitboards[7]);
  opponentattacks |= KingAttacks[opponentking];
  ourcaptures =
      KingAttacks[kingsquare] & ((~opponentattacks) & Bitboards[opposite]);
  int capturenumber = __builtin_popcountll(ourcaptures);
  int movenumber;
  for (int i = 0; i < capturenumber; i++) {
    int capturesquare = __builtin_ctzll(ourcaptures);
    int captured = pieces[capturesquare] % 8;
    int notation = kingsquare | (capturesquare << 6);
    notation |= (color << 12);
    notation |= (7 << 13);
    notation |= (1 << 16);
    notation |= (captured << 17);
    movelist[movecount] = notation;
    movecount++;
    ourcaptures ^= (1ULL << capturesquare);
  }
  if (!capturesonly) {
    ourmoves = KingAttacks[kingsquare] & ((~opponentattacks) & (~preoccupied));
    movenumber = __builtin_popcountll(ourmoves);
    for (int i = 0; i < movenumber; i++) {
      int movesquare = __builtin_ctzll(ourmoves);
      int notation = kingsquare | (movesquare << 6);
      notation |= (color << 12);
      notation |= (7 << 13);
      movelist[movecount] = notation;
      movecount++;
      ourmoves ^= (1ULL << movesquare);
    }
  }
  U64 checks = checkers(color);
  if (__builtin_popcountll(checks) > 1) {
    return movecount;
  } else if (checks) {
    checkmask |= checks;
  } else {
    checkmask = ~(0ULL);
  }
  U64 ourpawns = Bitboards[color] & Bitboards[2];
  U64 ouralfils = Bitboards[color] & Bitboards[3];
  U64 ourferzes = Bitboards[color] & Bitboards[4];
  U64 ourknights = Bitboards[color] & Bitboards[5];
  U64 ourrooks = Bitboards[color] & Bitboards[6];
  pawncount = __builtin_popcountll(ourpawns);
  alfilcount = __builtin_popcountll(ouralfils);
  ferzcount = __builtin_popcountll(ourferzes);
  knightcount = __builtin_popcountll(ourknights);
  rookcount = __builtin_popcountll(ourrooks);
  for (int i = 0; i < pawncount; i++) {
    int pawnsquare = __builtin_ctzll(ourpawns);
    if ((pinnedpieces & (1ULL << pawnsquare)) &&
        ((pawnsquare & 56) == pinrank)) {
      ourpawns ^= (1ULL << pawnsquare);
      continue;
    } else if ((pinnedpieces & (1ULL << pawnsquare)) && capturesonly) {
      ourpawns ^= (1ULL << pawnsquare);
      continue;
    }
    int capturenumber = 0;
    if ((pinnedpieces & (1ULL << pawnsquare)) == 0ULL) {
      ourcaptures = PawnAttacks[color][pawnsquare] & Bitboards[opposite];
      ourcaptures &= checkmask;
      capturenumber = __builtin_popcountll(ourcaptures);
    }
    for (int j = 0; j < capturenumber; j++) {
      int capturesquare = __builtin_ctzll(ourcaptures);
      int captured = pieces[capturesquare] % 8;
      int notation = pawnsquare | (capturesquare << 6);
      notation |= (color << 12);
      notation |= (2 << 13);
      notation |= (1 << 16);
      notation |= (captured << 17);
      if (((color == 0) && (capturesquare & 56) == 56) ||
          ((color == 1) && (capturesquare & 56) == 0)) {
        movelist[movecount] = notation | (1 << 20);
        movecount++;
      } else {
        movelist[movecount] = notation;
        movecount++;
      }
      ourcaptures ^= (1ULL << capturesquare);
    }
    if (!capturesonly) {
      ourmoves = (1ULL << (pawnsquare + 8 * (1 - 2 * color))) & (~preoccupied);
      ourmoves &= checkmask;
      int movenumber = __builtin_popcountll(ourmoves);
      for (int j = 0; j < movenumber; j++) {
        int movesquare = __builtin_ctzll(ourmoves);
        int notation = pawnsquare | (movesquare << 6);
        notation |= (color << 12);
        notation |= (2 << 13);
        if (((color == 0) && (movesquare & 56) == 56) ||
            ((color == 1) && (movesquare & 56) == 0)) {
          movelist[movecount] = notation | (1 << 20);
          movecount++;
        } else {
          movelist[movecount] = notation;
          movecount++;
        }
        ourmoves ^= (1ULL << movesquare);
      }
    }
    ourpawns ^= (1ULL << pawnsquare);
  }
  for (int i = 0; i < alfilcount; i++) {
    int alfilsquare = __builtin_ctzll(ouralfils);
    if (pinnedpieces & (1ULL << alfilsquare)) {
      ouralfils ^= (1ULL << alfilsquare);
      continue;
    }
    ourmask = AlfilAttacks[alfilsquare];
    ourmask &= checkmask;
    ourcaptures = ourmask & Bitboards[opposite];
    int capturenumber = __builtin_popcountll(ourcaptures);
    for (int j = 0; j < capturenumber; j++) {
      int capturesquare = __builtin_ctzll(ourcaptures);
      int captured = pieces[capturesquare] % 8;
      int notation = alfilsquare | (capturesquare << 6);
      notation |= (color << 12);
      notation |= (3 << 13);
      notation |= (1 << 16);
      notation |= (captured << 17);
      movelist[movecount] = notation;
      movecount++;
      ourcaptures ^= (1ULL << capturesquare);
    }
    if (!capturesonly) {
      ourmoves = ourmask & (~preoccupied);
      int movenumber = __builtin_popcountll(ourmoves);
      for (int j = 0; j < movenumber; j++) {
        int movesquare = __builtin_ctzll(ourmoves);
        int notation = alfilsquare | (movesquare << 6);
        notation |= (color << 12);
        notation |= (3 << 13);
        movelist[movecount] = notation;
        movecount++;
        ourmoves ^= (1ULL << movesquare);
      }
    }
    ouralfils ^= (1ULL << alfilsquare);
  }
  for (int i = 0; i < ferzcount; i++) {
    int ferzsquare = __builtin_ctzll(ourferzes);
    if (pinnedpieces & (1ULL << ferzsquare)) {
      ourferzes ^= (1ULL << ferzsquare);
      continue;
    }
    ourmask = FerzAttacks[ferzsquare];
    ourmask &= checkmask;
    ourcaptures = ourmask & Bitboards[opposite];
    int capturenumber = __builtin_popcountll(ourcaptures);
    for (int j = 0; j < capturenumber; j++) {
      int capturesquare = __builtin_ctzll(ourcaptures);
      int captured = pieces[capturesquare] % 8;
      int notation = ferzsquare | (capturesquare << 6);
      notation |= (color << 12);
      notation |= (4 << 13);
      notation |= (1 << 16);
      notation |= (captured << 17);
      movelist[movecount] = notation;
      movecount++;
      ourcaptures ^= (1ULL << capturesquare);
    }
    if (!capturesonly) {
      ourmoves = ourmask & (~preoccupied);
      int movenumber = __builtin_popcountll(ourmoves);
      for (int j = 0; j < movenumber; j++) {
        int movesquare = __builtin_ctzll(ourmoves);
        int notation = ferzsquare | (movesquare << 6);
        notation |= (color << 12);
        notation |= (4 << 13);
        movelist[movecount] = notation;
        movecount++;
        ourmoves ^= (1ULL << movesquare);
      }
    }
    ourferzes ^= (1ULL << ferzsquare);
  }
  for (int i = 0; i < knightcount; i++) {
    int knightsquare = __builtin_ctzll(ourknights);
    if (pinnedpieces & (1ULL << knightsquare)) {
      ourknights ^= (1ULL << knightsquare);
      continue;
    }
    ourmask = KnightAttacks[knightsquare];
    ourmask &= checkmask;
    ourcaptures = ourmask & Bitboards[opposite];
    int capturenumber = __builtin_popcountll(ourcaptures);
    for (int j = 0; j < capturenumber; j++) {
      int capturesquare = __builtin_ctzll(ourcaptures);
      int captured = pieces[capturesquare] % 8;
      int notation = knightsquare | (capturesquare << 6);
      notation |= (color << 12);
      notation |= (5 << 13);
      notation |= (1 << 16);
      notation |= (captured << 17);
      movelist[movecount] = notation;
      movecount++;
      ourcaptures ^= (1ULL << capturesquare);
    }
    if (!capturesonly) {
      ourmoves = ourmask & (~preoccupied);
      int movenumber = __builtin_popcountll(ourmoves);
      for (int j = 0; j < movenumber; j++) {
        int movesquare = __builtin_ctzll(ourmoves);
        int notation = knightsquare | (movesquare << 6);
        notation |= (color << 12);
        notation |= (5 << 13);
        movelist[movecount] = notation;
        movecount++;
        ourmoves ^= (1ULL << movesquare);
      }
    }
    ourknights ^= (1ULL << knightsquare);
  }
  for (int i = 0; i < rookcount; i++) {
    int rooksquare = __builtin_ctzll(ourrooks);
    ourmask = (GetRankAttacks(preoccupied, rooksquare) |
               FileAttacks(preoccupied, rooksquare));
    U64 pinmask = ~(0ULL);
    if (pinnedpieces & (1ULL << rooksquare)) {
      int rookrank = rooksquare & 56;
      if (rookrank == pinrank) {
        pinmask = GetRankAttacks(preoccupied, rooksquare);
      } else {
        pinmask = FileAttacks(preoccupied, rooksquare);
      }
    }
    ourmask &= (pinmask & checkmask);
    ourcaptures = ourmask & Bitboards[opposite];
    int capturenumber = __builtin_popcountll(ourcaptures);
    for (int j = 0; j < capturenumber; j++) {
      int capturesquare = __builtin_ctzll(ourcaptures);
      int captured = pieces[capturesquare] % 8;
      int notation = rooksquare | (capturesquare << 6);
      notation |= (color << 12);
      notation |= (6 << 13);
      notation |= (1 << 16);
      notation |= (captured << 17);
      movelist[movecount] = notation;
      movecount++;
      ourcaptures ^= (1ULL << capturesquare);
    }
    if (!capturesonly) {
      ourmoves = ourmask & (~preoccupied);
      int movenumber = __builtin_popcountll(ourmoves);
      for (int j = 0; j < movenumber; j++) {
        int movesquare = __builtin_ctzll(ourmoves);
        int notation = rooksquare | (movesquare << 6);
        notation |= (color << 12);
        notation |= (6 << 13);
        movelist[movecount] = notation;
        movecount++;
        ourmoves ^= (1ULL << movesquare);
      }
    }
    ourrooks ^= (1ULL << rooksquare);
  }
  return movecount;
}
U64 Position::perft(int depth, int initialdepth, int color) {
  int moves[maxmoves];
  int movcount = generatemoves(color, 0, moves);
  U64 ans = 0;
  if (depth > 1) {
    for (int i = 0; i < movcount; i++) {
      makemove(moves[i], true);
      if (depth == initialdepth) {
        std::cout << algebraic(moves[i]);
        std::cout << ": ";
      }
      ans += perft(depth - 1, initialdepth, color ^ 1);
      unmakemove(moves[i]);
    }
    if (depth == initialdepth - 1) {
      std::cout << ans << " ";
    }
    if (depth == initialdepth) {
      std::cout << "\n" << ans << "\n";
    }
    return ans;
  } else {
    if (initialdepth == 2) {
      std::cout << movcount << " ";
    }
    return movcount;
  }
}
U64 Position::perftnobulk(int depth, int initialdepth, int color) {
  int moves[maxmoves];
  int movcount = generatemoves(color, 0, moves);
  U64 ans = 0;
  for (int i = 0; i < movcount; i++) {
    makemove(moves[i], true);
    if (depth == initialdepth) {
      std::cout << algebraic(moves[i]);
      std::cout << ": ";
    }
    if (depth > 1) {
      ans += perftnobulk(depth - 1, initialdepth, color ^ 1);
    } else {
      ans++;
    }
    unmakemove(moves[i]);
  }
  if (depth == initialdepth - 1) {
    std::cout << ans << " ";
  }
  if (depth == initialdepth) {
    std::cout << "\n" << ans << "\n";
  }
  return ans;
}
void Position::parseFEN(std::string FEN) {
  gamelength = 0;
  root = 0;
  int progress = 0;
  for (int i = 0; i < 8; i++) {
    Bitboards[i] = 0ULL;
  }
  int tracker = 0;
  int castling = 0;
  int color = 0;
  while (FEN[tracker] != ' ') {
    char hm = FEN[tracker];
    if ('0' <= hm && hm <= '9') {
      int repeat = (int)hm - 48;
      for (int i = 0; i < repeat; i++) {
        pieces[(56 ^ progress)] = 0;
        progress++;
      }
    }
    if ('A' <= hm && hm <= 'Z') {
      Bitboards[0] |= (1ULL << (56 ^ progress));
      if (hm == 'P') {
        Bitboards[2] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 2;
      }
      if (hm == 'A' || hm == 'B') {
        Bitboards[3] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 3;
      }
      if (hm == 'F' || hm == 'Q') {
        Bitboards[4] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 4;
      }
      if (hm == 'N') {
        Bitboards[5] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 5;
      }
      if (hm == 'R') {
        Bitboards[6] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 6;
      }
      if (hm == 'K') {
        Bitboards[7] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 7;
      }
      progress++;
    }
    if ('a' <= hm && hm <= 'z') {
      Bitboards[1] |= (1ULL << (56 ^ progress));
      if (hm == 'p') {
        Bitboards[2] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 10;
      }
      if (hm == 'a' || hm == 'b') {
        Bitboards[3] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 11;
      }
      if (hm == 'f' || hm == 'q') {
        Bitboards[4] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 12;
      }
      if (hm == 'n') {
        Bitboards[5] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 13;
      }
      if (hm == 'r') {
        Bitboards[6] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 14;
      }
      if (hm == 'k') {
        Bitboards[7] |= (1ULL << (56 ^ progress));
        pieces[(56 ^ progress)] = 15;
      }
      progress++;
    }
    tracker++;
  }
  while (FEN[tracker] == ' ') {
    tracker++;
  }
  if (FEN[tracker] == 'b') {
    color = 1;
  }
  auxinfo = color;
  tracker += 6;
  int halfmove = (int)(FEN[tracker]) - 48;
  tracker++;
  if (FEN[tracker] != ' ') {
    halfmove = 10 * halfmove + (int)(FEN[tracker]) - 48;
  }
  auxinfo |= (halfmove << 1);
  zobristhash = scratchzobrist();
  zobrist[0] = zobristhash;
  history[0] = auxinfo;
}
std::string Position::getFEN() {
  std::string FEN = "";
  int empt = 0;
  char convert[2][6] = {{'P', 'B', 'Q', 'N', 'R', 'K'},
                        {'p', 'b', 'q', 'n', 'r', 'k'}};
  int color;
  int piece;
  for (int i = 0; i < 64; i++) {
    color = -1;
    for (int j = 0; j < 2; j++) {
      if (Bitboards[j] & (1ULL << (56 ^ i))) {
        color = j;
      }
    }
    if (color >= 0) {
      if (empt > 0) {
        FEN = FEN + (char)(empt + 48);
        empt = 0;
      }
      for (int j = 0; j < 6; j++) {
        if (Bitboards[j + 2] & (1ULL << (56 ^ i))) {
          piece = j;
        }
      }
      FEN = FEN + (convert[color][piece]);
    } else {
      empt++;
      if ((i & 7) == 7) {
        FEN = FEN + (char)(empt + 48);
        empt = 0;
      }
    }
    if (((i & 7) == 7) && (i < 63)) {
      FEN = FEN + '/';
    }
  }
  FEN = FEN + ' ';
  if (auxinfo & 1) {
    FEN = FEN + "b - - ";
  } else {
    FEN = FEN + "w - - ";
  }
  int halfmove = auxinfo >> 1;
  std::string bruh = "";
  do {
    bruh = bruh + (char)(halfmove % 10 + 48);
    halfmove /= 10;
  } while (halfmove > 0);
  reverse(bruh.begin(), bruh.end());
  FEN = FEN + bruh + " 1";
  return FEN;
}
bool Position::see_exceeds(int mov, int color, int threshold) {
  int see_values[6] = {100, 100, 170, 370, 640, 20000};
  int target = (mov >> 6) & 63;
  int victim = (mov >> 17) & 7;
  int attacker = (mov >> 13) & 7;

  int value = (victim > 0) ? see_values[victim - 2] - threshold : -threshold;
  if (value < 0) {
    return false;
  }
  if (attacker == 7 || value - see_values[attacker - 2] >= 0) {
    return true;
  }

  U64 occupied = (Bitboards[0] | Bitboards[1]) ^ (1ULL << (mov & 63));
  U64 us = Bitboards[color] & occupied;
  U64 enemy = Bitboards[color ^ 1];

  bool ourturn = false;
  int piececountus[6] = {-1, -1, -1, -1, -1, -1};
  int piececountopp[6] = {-1, -1, -1, -1, -1, -1};
  int currpieceus = attacker - 2;
  int currpieceopp = 0;
  int nextpieceus = 0;
  int nextpieceopp = 0;
  while (true) {
    if (ourturn) {
      bool end = false;
      while (!end) {
        if (nextpieceus > 5) {
          return (value >= 0);
        }
        if (piececountus[nextpieceus] < 0) {
          switch (nextpieceus) {
          case 0:
            piececountus[nextpieceus] = __builtin_popcountll(
                PawnAttacks[color ^ 1][target] & Bitboards[2] & us);
            break;
          case 1:
            piececountus[nextpieceus] =
                __builtin_popcountll(AlfilAttacks[target] & Bitboards[3] & us);
            break;
          case 2:
            piececountus[nextpieceus] =
                __builtin_popcountll(FerzAttacks[target] & Bitboards[4] & us);
            break;
          case 3:
            piececountus[nextpieceus] =
                __builtin_popcountll(KnightAttacks[target] & Bitboards[5] & us);
            break;
          case 4:
            piececountus[nextpieceus] = __builtin_popcountll(
                (FileAttacks(occupied & ~(Bitboards[6] & us), target) |
                 GetRankAttacks(occupied & ~(Bitboards[6] & us), target)) &
                Bitboards[6] & us);
            break;
          case 5:
            piececountus[nextpieceus] =
                __builtin_popcountll(KingAttacks[target] & Bitboards[7] & us);
          }
        }
        if (piececountus[nextpieceus] == 0) {
          nextpieceus++;
        } else {
          value += see_values[currpieceopp];
          if (value - see_values[nextpieceus] >= 0) {
            return true;
          }
          piececountus[nextpieceus]--;
          currpieceus = nextpieceus;
          end = true;
        }
      }
    } else {
      bool end = false;
      while (!end) {
        if (nextpieceopp > 5) {
          return (value >= 0);
        }
        if (piececountopp[nextpieceopp] < 0) {
          switch (nextpieceopp) {
          case 0:
            piececountopp[nextpieceopp] = __builtin_popcountll(
                PawnAttacks[color][target] & Bitboards[2] & enemy);
            break;
          case 1:
            piececountopp[nextpieceopp] = __builtin_popcountll(
                AlfilAttacks[target] & Bitboards[3] & enemy);
            break;
          case 2:
            piececountopp[nextpieceopp] = __builtin_popcountll(
                FerzAttacks[target] & Bitboards[4] & enemy);
            break;
          case 3:
            piececountopp[nextpieceopp] = __builtin_popcountll(
                KnightAttacks[target] & Bitboards[5] & enemy);
            break;
          case 4:
            piececountopp[nextpieceopp] = __builtin_popcountll(
                (FileAttacks(occupied & ~(Bitboards[6] & enemy), target) |
                 GetRankAttacks(occupied & ~(Bitboards[6] & enemy), target)) &
                Bitboards[6] & enemy);
            break;
          case 5:
            piececountopp[nextpieceopp] = __builtin_popcountll(
                KingAttacks[target] & Bitboards[7] & enemy);
          }
        }
        if (piececountopp[nextpieceopp] == 0) {
          nextpieceopp++;
        } else {
          value -= see_values[currpieceus];
          if (value + see_values[nextpieceopp] < 0) {
            return false;
          }
          piececountopp[nextpieceopp]--;
          currpieceopp = nextpieceopp;
          end = true;
        }
      }
    }
    ourturn = !ourturn;
  }
}