#include "position.h"
#include "consts.h"
#include <bit>
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
U64 bitboard(int square) { return 1ULL << square; }
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
          attacks |= bitboard(k);
          if (bitboard(k) & occupied) {
            k = 0;
          }
          k--;
        }
      }
      if (j < 7) {
        int k = j + 1;
        while (k <= 7) {
          attacks |= bitboard(k);
          if (bitboard(k) & occupied) {
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
  U64 backwards = std::byteswap(forwards);
  forwards = forwards - 2 * bitboard(square);
  backwards = backwards - 2 * bitboard((56 ^ square));
  backwards = std::byteswap(backwards);
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
std::string algebraic(Move mov) {
  std::string convert[64] = {
      "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1", "a2", "b2", "c2",
      "d2", "e2", "f2", "g2", "h2", "a3", "b3", "c3", "d3", "e3", "f3",
      "g3", "h3", "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4", "a5",
      "b5", "c5", "d5", "e5", "f5", "g5", "h5", "a6", "b6", "c6", "d6",
      "e6", "f6", "g6", "h6", "a7", "b7", "c7", "d7", "e7", "f7", "g7",
      "h7", "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"};
  std::string header = convert[mov.from()] + convert[mov.to()];
  if (mov.promoted()) {
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
int perspectivepiece(Piece p, bool stm) {
  if (p.empty()) {
    return 0;
  } else {
    return p.type() + ((p.color() == stm) ? 0 : 6) - 1;
  }
}
U64 Position::scratchzobrist() {
  U64 scratch = 0ULL;
  for (int i = 0; i < 64; i++) {
    Piece piece = pieces[i];
    if (piece.type() > 0) {
      scratch ^= hashes[piece.color()][i];
      scratch ^= hashes[piece.type()][i];
    }
  }
  if (stm) {
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
  stm = 0;
  PieceType startpiece[16] = {Rook,   Knight, Alfil, King, Ferz, Alfil,
                              Knight, Rook,   Pawn,  Pawn, Pawn, Pawn,
                              Pawn,   Pawn,   Pawn,  Pawn};
  for (int i = 0; i < 16; i++) {
    pieces[i] = Piece(White, startpiece[i]);
    pieces[56 ^ i] = Piece(Black, startpiece[i]);
  }
  for (int i = 16; i < 48; i++) {
    pieces[i] = Piece();
  }
  zobristhash = scratchzobrist();
}
bool Position::twokings() {
  return (Bitboards[White] | Bitboards[Black]) == Bitboards[King];
}
bool Position::bareking(bool color) {
  return (Bitboards[color] & Bitboards[King]) == Bitboards[color];
}
U64 Position::checkers(int color) {
  int kingsquare = std::countr_zero(Bitboards[color] & Bitboards[King]);
  int opposite = color ^ 1;
  U64 attacks = 0ULL;
  U64 occupied = Bitboards[White] | Bitboards[Black];
  attacks |= (KnightAttacks[kingsquare] & Bitboards[Knight]);
  attacks |= (PawnAttacks[color][kingsquare] & Bitboards[Pawn]);
  attacks |= (AlfilAttacks[kingsquare] & Bitboards[Alfil]);
  attacks |= (FerzAttacks[kingsquare] & Bitboards[Ferz]);
  attacks |= (GetRankAttacks(occupied, kingsquare) & Bitboards[Rook]);
  attacks |= (FileAttacks(occupied, kingsquare) & Bitboards[Rook]);
  attacks &= Bitboards[opposite];
  return attacks;
}
void Position::makemove(Move mov) {
  int from = mov.from();
  int to = mov.to();
  U8 captured = mov.captured();
  PieceType type = pieces[from].type();

  Bitboards[stm] ^= (bitboard(from) | bitboard(to));
  Bitboards[type] ^= (bitboard(from) | bitboard(to));
  pieces[to] = pieces[from];
  pieces[from] = Piece();
  zobristhash ^= (hashes[stm][from] ^ hashes[stm][to]);
  zobristhash ^= (hashes[type][from] ^ hashes[type][to]);
  halfmovecount++;
  if (type == Pawn) {
    halfmovecount = 0;
  }
  if (captured != None) {
    Bitboards[!stm] ^= bitboard(to);
    Bitboards[captured] ^= bitboard(to);
    zobristhash ^= (hashes[!stm][to] ^ hashes[captured][to]);
    halfmovecount = 0;
  }
  if (mov.promoted()) {
    Bitboards[Pawn] ^= bitboard(to);
    Bitboards[Ferz] ^= bitboard(to);
    pieces[to] = Piece(stm, Ferz);
    zobristhash ^= (hashes[Pawn][to] ^ hashes[Ferz][to]);
  }
  stm = !stm;
  zobristhash ^= colorhash;
}
void Position::unmakemove(Move mov) {
  stm = !stm;
  int from = mov.from();
  int to = mov.to();
  PieceType captured = mov.captured();
  PieceType type = pieces[to].type();

  Bitboards[stm] ^= (bitboard(from) | bitboard(to));
  Bitboards[type] ^= (bitboard(from) | bitboard(to));
  pieces[from] = pieces[to];
  pieces[to] = Piece();
  if (captured != None) {
    Bitboards[!stm] ^= bitboard(to);
    Bitboards[captured] ^= bitboard(to);
    pieces[to] = Piece(!stm, captured);
  }
  if (mov.promoted()) {
    Bitboards[Pawn] ^= bitboard(from);
    Bitboards[Ferz] ^= bitboard(from);
    pieces[from] = Piece(stm, Pawn);
  }
}
int Position::generatemoves(Move *movelist) {
  int movecount = 0;
  int kingsquare = std::countr_zero(Bitboards[stm] & Bitboards[King]);
  int pinrank = kingsquare & 56;
  int pinfile = kingsquare & 7;
  U64 opponentattacks = 0ULL;
  U64 pinnedpieces = 0ULL;
  U64 checkmask = 0ULL;
  U64 preoccupied = Bitboards[White] | Bitboards[Black];
  U64 kingRank = GetRankAttacks(preoccupied, kingsquare);
  U64 kingFile = FileAttacks(preoccupied, kingsquare);
  U64 occupied = preoccupied ^ bitboard(kingsquare);
  U64 opponentpawns = Bitboards[!stm] & Bitboards[Pawn];
  U64 opponentalfils = Bitboards[!stm] & Bitboards[Alfil];
  U64 opponentferzes = Bitboards[!stm] & Bitboards[Ferz];
  U64 opponentknights = Bitboards[!stm] & Bitboards[Knight];
  U64 opponentrooks = Bitboards[!stm] & Bitboards[Rook];
  int pawncount = std::popcount(opponentpawns);
  int alfilcount = std::popcount(opponentalfils);
  int ferzcount = std::popcount(opponentferzes);
  int knightcount = std::popcount(opponentknights);
  int rookcount = std::popcount(opponentrooks);
  U64 ourmoves = 0ULL;
  U64 ourmask = 0ULL;
  for (int i = 0; i < pawncount; i++) {
    int pawnsquare = std::countr_zero(opponentpawns);
    opponentattacks |= PawnAttacks[!stm][pawnsquare];
    opponentpawns ^= bitboard(pawnsquare);
  }
  U64 opponentpawnattacks = opponentattacks;
  for (int i = 0; i < alfilcount; i++) {
    int alfilsquare = std::countr_zero(opponentalfils);
    opponentattacks |= AlfilAttacks[alfilsquare];
    opponentalfils ^= bitboard(alfilsquare);
  }
  U64 opponentalfilattacks = opponentattacks;
  for (int i = 0; i < ferzcount; i++) {
    int ferzsquare = std::countr_zero(opponentferzes);
    opponentattacks |= FerzAttacks[ferzsquare];
    opponentferzes ^= bitboard(ferzsquare);
  }
  U64 opponentferzattacks = opponentattacks;
  for (int i = 0; i < knightcount; i++) {
    int knightsquare = std::countr_zero(opponentknights);
    opponentattacks |= KnightAttacks[knightsquare];
    opponentknights ^= bitboard(knightsquare);
  }
  U64 opponentknightattacks = opponentattacks;
  for (int i = 0; i < rookcount; i++) {
    int rooksquare = std::countr_zero(opponentrooks);
    U64 r = GetRankAttacks(occupied, rooksquare);
    U64 file = FileAttacks(occupied, rooksquare);
    if (!(r & bitboard(kingsquare))) {
      pinnedpieces |= (r & kingRank);
    } else {
      checkmask |= (GetRankAttacks(preoccupied, rooksquare) & kingRank);
    }
    if (!(file & bitboard(kingsquare))) {
      pinnedpieces |= (file & kingFile);
    } else {
      checkmask |= (FileAttacks(preoccupied, rooksquare) & kingFile);
    }
    opponentattacks |= (r | file);
    opponentrooks ^= bitboard(rooksquare);
  }
  int opponentking = std::countr_zero(Bitboards[!stm] & Bitboards[King]);
  opponentattacks |= KingAttacks[opponentking];
  ourmoves = KingAttacks[kingsquare] & ((~opponentattacks) & (~Bitboards[stm]));
  int movenumber = std::popcount(ourmoves);
  for (int i = 0; i < movenumber; i++) {
    int movesquare = std::countr_zero(ourmoves);
    PieceType captured = pieces[movesquare].type();
    movelist[movecount] = Move(kingsquare, movesquare, false, captured);
    movecount++;
    ourmoves &= (ourmoves - 1);
  }
  U64 checks = checkers(stm);
  if (std::popcount(checks) > 1) {
    return movecount;
  } else if (checks) {
    checkmask |= checks;
  } else {
    checkmask = ~(0ULL);
  }
  U64 movemask = checkmask & (~Bitboards[stm]);
  U64 ourpawns = Bitboards[stm] & Bitboards[Pawn];
  U64 ouralfils = Bitboards[stm] & Bitboards[Alfil];
  U64 ourferzes = Bitboards[stm] & Bitboards[Ferz];
  U64 ourknights = Bitboards[stm] & Bitboards[Knight];
  U64 ourrooks = Bitboards[stm] & Bitboards[Rook];
  pawncount = std::popcount(ourpawns);
  alfilcount = std::popcount(ouralfils);
  ferzcount = std::popcount(ourferzes);
  knightcount = std::popcount(ourknights);
  rookcount = std::popcount(ourrooks);
  for (int i = 0; i < pawncount; i++) {
    int pawnsquare = std::countr_zero(ourpawns);
    if ((pinnedpieces & bitboard(pawnsquare)) &&
        ((pawnsquare & 56) == pinrank)) {
      ourpawns &= (ourpawns - 1);
      continue;
    }
    ourmoves = bitboard((pawnsquare + 8 * (1 - 2 * stm))) & (~preoccupied);
    if ((pinnedpieces & bitboard(pawnsquare)) == 0ULL) {
      ourmoves |= (PawnAttacks[stm][pawnsquare] & Bitboards[!stm]);
    }
    ourmoves &= checkmask;
    movenumber = std::popcount(ourmoves);
    for (int i = 0; i < movenumber; i++) {
      int movesquare = std::countr_zero(ourmoves);
      PieceType captured = pieces[movesquare].type();
      bool promo =
          (!stm && (movesquare & 56) == 56) || (stm && (movesquare & 56) == 0);
      movelist[movecount] = Move(pawnsquare, movesquare, promo, captured);
      movecount++;
      ourmoves &= (ourmoves - 1);
    }
    ourpawns &= (ourpawns - 1);
  }
  for (int i = 0; i < alfilcount; i++) {
    int alfilsquare = std::countr_zero(ouralfils);
    if (pinnedpieces & bitboard(alfilsquare)) {
      ouralfils ^= bitboard(alfilsquare);
      continue;
    }
    ourmask = AlfilAttacks[alfilsquare];
    ourmoves = ourmask & movemask;
    int movenumber = std::popcount(ourmoves);
    for (int i = 0; i < movenumber; i++) {
      int movesquare = std::countr_zero(ourmoves);
      PieceType captured = pieces[movesquare].type();
      movelist[movecount] = Move(alfilsquare, movesquare, false, captured);
      movecount++;
      ourmoves &= (ourmoves - 1);
    }
    ouralfils &= (ouralfils - 1);
  }
  for (int i = 0; i < ferzcount; i++) {
    int ferzsquare = std::countr_zero(ourferzes);
    if (pinnedpieces & bitboard(ferzsquare)) {
      ourferzes ^= bitboard(ferzsquare);
      continue;
    }
    ourmask = FerzAttacks[ferzsquare];
    ourmoves = ourmask & movemask;
    int movenumber = std::popcount(ourmoves);
    for (int i = 0; i < movenumber; i++) {
      int movesquare = std::countr_zero(ourmoves);
      PieceType captured = pieces[movesquare].type();
      movelist[movecount] = Move(ferzsquare, movesquare, false, captured);
      movecount++;
      ourmoves &= (ourmoves - 1);
    }
    ourferzes &= (ourferzes - 1);
  }
  for (int i = 0; i < knightcount; i++) {
    int knightsquare = std::countr_zero(ourknights);
    if (pinnedpieces & bitboard(knightsquare)) {
      ourknights ^= bitboard(knightsquare);
      continue;
    }
    ourmask = KnightAttacks[knightsquare];
    ourmoves = ourmask & movemask;
    int movenumber = std::popcount(ourmoves);
    for (int i = 0; i < movenumber; i++) {
      int movesquare = std::countr_zero(ourmoves);
      PieceType captured = pieces[movesquare].type();
      movelist[movecount] = Move(knightsquare, movesquare, false, captured);
      movecount++;
      ourmoves &= (ourmoves - 1);
    }
    ourknights &= (ourknights - 1);
  }
  for (int i = 0; i < rookcount; i++) {
    int rooksquare = std::countr_zero(ourrooks);
    ourmask = (GetRankAttacks(preoccupied, rooksquare) |
               FileAttacks(preoccupied, rooksquare));
    U64 pinmask = ~(0ULL);
    if (pinnedpieces & bitboard(rooksquare)) {
      int rookrank = rooksquare & 56;
      if (rookrank == pinrank) {
        pinmask = GetRankAttacks(preoccupied, rooksquare);
      } else {
        pinmask = FileAttacks(preoccupied, rooksquare);
      }
    }
    ourmoves = ourmask & movemask & pinmask;
    int movenumber = std::popcount(ourmoves);
    for (int i = 0; i < movenumber; i++) {
      int movesquare = std::countr_zero(ourmoves);
      PieceType captured = pieces[movesquare].type();
      movelist[movecount] = Move(rooksquare, movesquare, false, captured);
      movecount++;
      ourmoves &= (ourmoves - 1);
    }
    ourrooks &= (ourrooks - 1);
  }
  return movecount;
}
U64 Position::perft(int depth, int initialdepth) {
  Move moves[maxmoves];
  int movcount = generatemoves(moves);
  U64 ans = 0;
  if (depth > 1) {
    for (int i = 0; i < movcount; i++) {
      makemove(moves[i]);
      if (depth == initialdepth) {
        std::cout << algebraic(moves[i]);
        std::cout << ": ";
      }
      ans += perft(depth - 1, initialdepth);
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
U64 Position::perftnobulk(int depth, int initialdepth) {
  Move moves[maxmoves];
  int movcount = generatemoves(moves);
  U64 ans = 0;
  for (int i = 0; i < movcount; i++) {
    makemove(moves[i]);
    if (depth == initialdepth) {
      std::cout << algebraic(moves[i]);
      std::cout << ": ";
    }
    if (depth > 1) {
      ans += perftnobulk(depth - 1, initialdepth);
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
  int progress = 0;
  for (int i = 0; i < 8; i++) {
    Bitboards[i] = 0ULL;
  }
  int tracker = 0;
  int castling = 0;
  stm = 0;
  PieceType types[18] = {Alfil, Alfil,  None, None, Alfil, Ferz,
                         Ferz,  None,   None, None, King,  None,
                         None,  Knight, None, Pawn, Ferz,  Rook};
  while (FEN[tracker] != ' ') {
    char hm = FEN[tracker];
    bool color;
    PieceType type = None;
    if ('0' <= hm && hm <= '9') {
      int repeat = hm - '0';
      for (int i = 0; i < repeat; i++) {
        pieces[(56 ^ progress)] = Piece();
        progress++;
      }
    }
    if ('A' <= hm && hm <= 'R') {
      color = White;
      type = types[hm - 'A'];
    }
    if ('a' <= hm && hm <= 'r') {
      color = Black;
      type = types[hm - 'a'];
    }
    if (type != None) {
      Bitboards[color] ^= bitboard((56 ^ progress));
      Bitboards[type] ^= bitboard((56 ^ progress));
      pieces[(56 ^ progress)] = Piece(color, type);
      progress++;
    }
    tracker++;
  }
  while (FEN[tracker] == ' ') {
    tracker++;
  }
  if (FEN[tracker] == 'b') {
    stm = 1;
  }
  tracker += 6;
  halfmovecount = 0;
  while (FEN[tracker] != ' ') {
    halfmovecount = 10 * halfmovecount + (int)(FEN[tracker] - '0');
    tracker++;
  }
  zobristhash = scratchzobrist();
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
      if (Bitboards[j] & bitboard((56 ^ i))) {
        color = j;
      }
    }
    if (color >= 0) {
      if (empt > 0) {
        FEN = FEN + (char)(empt + 48);
        empt = 0;
      }
      for (int j = 0; j < 6; j++) {
        if (Bitboards[j + 2] & bitboard((56 ^ i))) {
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
  if (stm) {
    FEN = FEN + "b - - ";
  } else {
    FEN = FEN + "w - - ";
  }
  int halfmove = halfmovecount;
  std::string bruh = "";
  do {
    bruh = bruh + (char)(halfmove % 10 + 48);
    halfmove /= 10;
  } while (halfmove > 0);
  reverse(bruh.begin(), bruh.end());
  FEN = FEN + bruh + " 1";
  return FEN;
}