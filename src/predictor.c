//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include <string.h>
#include "predictor.h"

//
// Student Information
//
const char *studentName = "NAME";
const char *studentID   = "PID";
const char *email       = "xuz063@ucsd.edu";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = { "Static", "Gshare",
                          "Tournament", "Custom" };

int ghistoryBits; // Number of bits used for Global History
int lhistoryBits; // Number of bits used for Local History
int pcIndexBits;  // Number of bits used for PC index
int bpType;       // Branch Prediction Type
int verbose;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//

// Global history register, shared by G-Share and Tournament (only one
// predictor runs per process). The newest outcome is shifted into bit 0;
// initialized to all NOTTAKEN.
static uint32_t ghistory;
static uint32_t ghistoryMask;   // low ghistoryBits bits set

// G-Share: one table of 2^ghistoryBits 2-bit counters, indexed by
// (PC XOR global history).
static uint8_t *gshareBHT;

// Tournament (Alpha 21264 style, every counter is 2-bit):
//   global predictor : 2^ghistoryBits counters, indexed by global history
//   choice predictor : 2^ghistoryBits counters, indexed by global history;
//                      a value >= WT selects the global predictor
//   local history    : 2^pcIndexBits entries of lhistoryBits bits, indexed by PC
//   local predictor  : 2^lhistoryBits counters, indexed by local history
static uint8_t  *tourGlobalBHT;
static uint8_t  *tourChoice;
static uint32_t *tourLocalHist;
static uint8_t  *tourLocalBHT;
static uint32_t  lhistoryMask;  // low lhistoryBits bits set
static uint32_t  pcIndexMask;   // low pcIndexBits bits set

// Custom: TAGE (TAgged GEometric history length predictor).
//
// I implemented TAGE as my custom predictor. A small bimodal table gives the
// default prediction; seven partially tagged tables are indexed by hashes of
// the PC with global histories of geometrically increasing length
// (4, 8, 15, 28, 52, 100, 190 branches). The prediction comes from the hit
// with the longest history (the "provider"); when the provider entry looks
// newly allocated, the next-longest hit ("alternate") may be used instead.
// On a misprediction a new entry is allocated in a longer-history table.
//
// Storage (bits of state the predictor keeps):
//   tageBase      4096 entries x 2-bit counter                       =  8192
//   tageTable[0]   512 entries x (3 ctr + 1 useful +  7 tag) = 11    =  5632
//   tageTable[1]   512 entries x (3 + 1 +  8) = 12                   =  6144
//   tageTable[2]   512 entries x (3 + 1 +  8) = 12                   =  6144
//   tageTable[3]   512 entries x (3 + 1 +  9) = 13                   =  6656
//   tageTable[4]   512 entries x (3 + 1 + 10) = 14                   =  7168
//   tageTable[5]   512 entries x (3 + 1 + 11) = 15                   =  7680
//   tageTable[6]   512 entries x (3 + 1 + 12) = 16                   =  8192
//                                                    tables subtotal = 55808
//   tageGhist     global history register, 3 x 64 bits               =   192
//   tagePathHist  path history (one PC bit per branch)               =    16
//   tageFold*     folded histories, per table 9 (index) + w + (w-1)
//                 (w = tag width): 22+24+24+26+28+30+32              =   186
//   tageUseAlt    4-bit counter: trust the alternate prediction?     =     4
//   tageTick      18-bit counter for the periodic useful-bit reset   =    18
//   tageRng       32-bit pseudo-random state for allocation          =    32
//                                                              total = 56256
// Limit is 64K + 256 = 65792 bits, so 9536 bits remain unused.
// (The C types are wider than the logical fields, e.g. a 3-bit counter is
// stored in an int8_t; the count above uses the logical widths.)
#define TAGE_NUM_TABLES   7
#define TAGE_BASE_LOG     12   // 4096-entry bimodal table
#define TAGE_TABLE_LOG    9    // 512 entries per tagged table
#define TAGE_HIST_WORDS   3    // 3 x 64 = 192 bits >= longest history + 1
#define TAGE_PATH_BITS    16
#define TAGE_CTR_MAX      3    // 3-bit signed counter: -4..3, >= 0 predicts TAKEN
#define TAGE_CTR_MIN      (-4)
#define TAGE_USE_ALT_MAX  15   // 4-bit counter
#define TAGE_USE_ALT_INIT 8
#define TAGE_U_RESET_LOG  18   // clear every useful bit each 2^18 branches
#define TAGE_RNG_SEED     12345u

static const int tageHistLen[] = { 4, 8, 15, 28, 52, 100, 190 };
static const int tageTagBits[] = { 7, 8, 8,  9, 10,  11,  12 };

typedef struct {
  int8_t   ctr;   // 3-bit signed prediction counter
  uint16_t tag;   // partial tag, tageTagBits[t] bits
  uint8_t  u;     // 1-bit useful flag
} TageEntry;

// A global history of histLen bits compressed (XOR-folded) to width bits,
// updated incrementally as each outcome enters and the oldest one leaves.
typedef struct {
  uint32_t value;
  int      width;
  int      histLen;
} FoldedHist;

static uint8_t    tageBase[1 << TAGE_BASE_LOG];
static TageEntry  tageTable[TAGE_NUM_TABLES][1 << TAGE_TABLE_LOG];
static uint64_t   tageGhist[TAGE_HIST_WORDS];   // bit k = outcome k branches ago
static uint32_t   tagePathHist;
static FoldedHist tageFoldIdx[TAGE_NUM_TABLES];
static FoldedHist tageFoldTag1[TAGE_NUM_TABLES];
static FoldedHist tageFoldTag2[TAGE_NUM_TABLES];
static uint8_t    tageUseAlt;
static uint32_t   tageTick;
static uint32_t   tageRng;

//------------------------------------//
//          Helper Functions          //
//------------------------------------//

// Direction predicted by a 2-bit saturating counter:
// SN/WN predict NOTTAKEN, WT/ST predict TAKEN.
static inline uint8_t
counter_predict(uint8_t counter)
{
  return (counter >= WT) ? TAKEN : NOTTAKEN;
}

// Next state of a 2-bit saturating counter.
// TAKEN moves SN -> WN -> WT -> ST, NOTTAKEN moves the other way;
// both saturate at the ends.
static inline uint8_t
counter_update(uint8_t counter, uint8_t outcome)
{
  if (outcome == TAKEN) {
    return (counter < ST) ? counter + 1 : ST;
  }
  return (counter > SN) ? counter - 1 : SN;
}

// Mask with the low 'bits' bits set (bits in 0..32).
static inline uint32_t
low_mask(int bits)
{
  return (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1);
}

// Allocate 'bytes' zero-filled bytes, or exit if memory runs out.
static void *
alloc_or_die(size_t bytes)
{
  void *p = calloc(bytes, 1);
  if (p == NULL) {
    fprintf(stderr, "predictor: out of memory\n");
    exit(1);
  }
  return p;
}

// Allocate 'entries' 2-bit counters, all initialized to 'value'.
static uint8_t *
alloc_counters(size_t entries, uint8_t value)
{
  uint8_t *table = (uint8_t *)alloc_or_die(entries);
  memset(table, value, entries);
  return table;
}

//------------------------------------//
//         G-Share Predictor          //
//------------------------------------//

// Storage: 2^ghistoryBits * 2 bits of counters + ghistoryBits of history.
// For --gshare:13 that is 8192 * 2 = 16384 bits (+13 history bits).

static void
gshare_init(void)
{
  gshareBHT    = alloc_counters((size_t)1 << ghistoryBits, WN);
  ghistoryMask = low_mask(ghistoryBits);
  ghistory     = 0;
}

// Index = low ghistoryBits bits of (PC XOR global history).
static inline uint32_t
gshare_index(uint32_t pc)
{
  return (pc ^ ghistory) & ghistoryMask;
}

static uint8_t
gshare_predict(uint32_t pc)
{
  return counter_predict(gshareBHT[gshare_index(pc)]);
}

// Train the counter that made the prediction, then shift the outcome
// into the global history.
static void
gshare_train(uint32_t pc, uint8_t outcome)
{
  uint32_t idx = gshare_index(pc);
  gshareBHT[idx] = counter_update(gshareBHT[idx], outcome);
  ghistory = ((ghistory << 1) | outcome) & ghistoryMask;
}

//------------------------------------//
//        Tournament Predictor        //
//------------------------------------//

// Storage: global 2^g*2 + choice 2^g*2 + local history 2^p*l + local 2^l*2
// bits (+ g bits of global history). For --tournament:9:10:10 that is
// 512*2 + 512*2 + 1024*10 + 1024*2 = 14336 bits (+9 history bits).

static void
tournament_init(void)
{
  size_t globalEntries = (size_t)1 << ghistoryBits;

  tourGlobalBHT = alloc_counters(globalEntries, WN);
  tourChoice    = alloc_counters(globalEntries, WT);  // weakly select global
  tourLocalBHT  = alloc_counters((size_t)1 << lhistoryBits, WN);
  tourLocalHist = (uint32_t *)alloc_or_die(((size_t)1 << pcIndexBits) *
                                           sizeof(uint32_t));  // all NOTTAKEN

  ghistoryMask = low_mask(ghistoryBits);
  lhistoryMask = low_mask(lhistoryBits);
  pcIndexMask  = low_mask(pcIndexBits);
  ghistory     = 0;
}

// Table indices and component predictions for one branch.
typedef struct {
  uint32_t globalIdx;   // global history: indexes the global BHT and choice
  uint32_t pcIdx;       // low PC bits: indexes the local history table
  uint32_t localIdx;    // this branch's local history: indexes the local BHT
  uint8_t  globalPred;
  uint8_t  localPred;
} TourLookup;

static TourLookup
tournament_lookup(uint32_t pc)
{
  TourLookup r;
  r.globalIdx  = ghistory & ghistoryMask;
  r.pcIdx      = pc & pcIndexMask;
  r.localIdx   = tourLocalHist[r.pcIdx] & lhistoryMask;
  r.globalPred = counter_predict(tourGlobalBHT[r.globalIdx]);
  r.localPred  = counter_predict(tourLocalBHT[r.localIdx]);
  return r;
}

static uint8_t
tournament_predict(uint32_t pc)
{
  TourLookup r = tournament_lookup(pc);
  return (tourChoice[r.globalIdx] >= WT) ? r.globalPred : r.localPred;
}

static void
tournament_train(uint32_t pc, uint8_t outcome)
{
  TourLookup r = tournament_lookup(pc);

  // 1. The choice counter learns only when the two components disagree:
  //    toward ST (global) if global was right, toward SN (local) otherwise.
  if (r.globalPred != r.localPred) {
    uint8_t globalRight = (r.globalPred == outcome) ? TAKEN : NOTTAKEN;
    tourChoice[r.globalIdx] = counter_update(tourChoice[r.globalIdx], globalRight);
  }

  // 2. Both components always train on the outcome.
  tourGlobalBHT[r.globalIdx] = counter_update(tourGlobalBHT[r.globalIdx], outcome);
  tourLocalBHT[r.localIdx]   = counter_update(tourLocalBHT[r.localIdx], outcome);

  // 3. Shift the outcome into this branch's local history and the global history.
  tourLocalHist[r.pcIdx] = ((r.localIdx << 1) | outcome) & lhistoryMask;
  ghistory = ((ghistory << 1) | outcome) & ghistoryMask;
}

//------------------------------------//
//     Custom Predictor (TAGE)        //
//------------------------------------//

// Storage breakdown: see the comment above the TAGE data structures.

static void
fold_init(FoldedHist *f, int width, int histLen)
{
  f->value   = 0;
  f->width   = width;
  f->histLen = histLen;
}

// Shift newBit into the folded history and remove outBit, the outcome that
// just became older than histLen branches.
static void
fold_update(FoldedHist *f, uint8_t newBit, uint8_t outBit)
{
  f->value  = (f->value << 1) | newBit;
  f->value ^= (uint32_t)outBit << (f->histLen % f->width);
  f->value ^= f->value >> f->width;
  f->value &= low_mask(f->width);
}

// Outcome of the branch 'age' branches ago (age 0 = most recent).
static inline uint8_t
tage_ghist_bit(int age)
{
  return (uint8_t)((tageGhist[age / 64] >> (age % 64)) & 1);
}

static void
tage_ghist_push(uint8_t outcome)
{
  int w;
  for (w = TAGE_HIST_WORDS - 1; w > 0; w--) {
    tageGhist[w] = (tageGhist[w] << 1) | (tageGhist[w - 1] >> 63);
  }
  tageGhist[0] = (tageGhist[0] << 1) | outcome;
}

// Next state of a 3-bit signed counter (-4..3).
static inline int8_t
tage_ctr_update(int8_t ctr, uint8_t outcome)
{
  if (outcome == TAKEN) {
    return (ctr < TAGE_CTR_MAX) ? ctr + 1 : TAGE_CTR_MAX;
  }
  return (ctr > TAGE_CTR_MIN) ? ctr - 1 : TAGE_CTR_MIN;
}

static void
tage_init(void)
{
  int t;
  memset(tageBase, WN, sizeof(tageBase));
  memset(tageTable, 0, sizeof(tageTable));
  memset(tageGhist, 0, sizeof(tageGhist));   // all NOTTAKEN
  for (t = 0; t < TAGE_NUM_TABLES; t++) {
    fold_init(&tageFoldIdx[t],  TAGE_TABLE_LOG,     tageHistLen[t]);
    fold_init(&tageFoldTag1[t], tageTagBits[t],     tageHistLen[t]);
    fold_init(&tageFoldTag2[t], tageTagBits[t] - 1, tageHistLen[t]);
  }
  tagePathHist = 0;
  tageUseAlt   = TAGE_USE_ALT_INIT;
  tageTick     = 0;
  tageRng      = TAGE_RNG_SEED;
}

// Index into tagged table t: PC hashed with the table's folded global
// history and a few bits of path history.
static uint32_t
tage_index(int t, uint32_t pc)
{
  int      pathBits = (tageHistLen[t] < TAGE_PATH_BITS) ? tageHistLen[t] : TAGE_PATH_BITS;
  uint32_t path     = tagePathHist & low_mask(pathBits);
  return (pc ^ (pc >> TAGE_TABLE_LOG) ^ tageFoldIdx[t].value ^ (path * (uint32_t)(t + 1)))
         & low_mask(TAGE_TABLE_LOG);
}

// Partial tag for table t: PC hashed with two differently sized foldings of
// the same history, so different histories rarely produce the same tag.
static uint16_t
tage_tag(int t, uint32_t pc)
{
  return (uint16_t)((pc ^ tageFoldTag1[t].value ^ (tageFoldTag2[t].value << 1))
                    & low_mask(tageTagBits[t]));
}

// Everything predict and train need to know about one branch.
typedef struct {
  uint32_t baseIdx;
  uint32_t idx[TAGE_NUM_TABLES];
  uint16_t tag[TAGE_NUM_TABLES];
  int      provider;      // longest-history table whose tag matches, -1 if none
  int      alt;           // next matching table below the provider, -1 if none
  uint8_t  providerPred;  // provider's prediction (bimodal if no provider)
  uint8_t  altPred;       // alternate's prediction (bimodal if no alternate)
  uint8_t  weakNew;       // provider entry is weak and not useful: likely new
  uint8_t  pred;          // final prediction
} TageLookup;

static TageLookup
tage_lookup(uint32_t pc)
{
  TageLookup r;
  uint8_t    basePred;
  int        t;

  r.baseIdx = pc & low_mask(TAGE_BASE_LOG);
  for (t = 0; t < TAGE_NUM_TABLES; t++) {
    r.idx[t] = tage_index(t, pc);
    r.tag[t] = tage_tag(t, pc);
  }

  r.provider = -1;
  r.alt      = -1;
  for (t = TAGE_NUM_TABLES - 1; t >= 0; t--) {
    if (tageTable[t][r.idx[t]].tag == r.tag[t]) {
      if (r.provider < 0) {
        r.provider = t;
      } else {
        r.alt = t;
        break;
      }
    }
  }

  basePred  = counter_predict(tageBase[r.baseIdx]);
  r.altPred = (r.alt >= 0)
              ? ((tageTable[r.alt][r.idx[r.alt]].ctr >= 0) ? TAKEN : NOTTAKEN)
              : basePred;
  r.providerPred = basePred;
  r.weakNew      = 0;
  r.pred         = basePred;

  if (r.provider >= 0) {
    const TageEntry *e = &tageTable[r.provider][r.idx[r.provider]];
    r.providerPred = (e->ctr >= 0) ? TAKEN : NOTTAKEN;
    r.weakNew      = (e->ctr == 0 || e->ctr == -1) && e->u == 0;
    r.pred         = (r.weakNew && tageUseAlt >= TAGE_USE_ALT_INIT)
                     ? r.altPred : r.providerPred;
  }
  return r;
}

static uint8_t
tage_predict(uint32_t pc)
{
  return tage_lookup(pc).pred;
}

// On a misprediction, try to allocate one entry in a table with a longer
// history than the provider. A random coin flip sometimes skips the first
// candidate so that new entries spread over several tables. If every
// candidate is useful, age them instead so a later allocation can succeed.
static void
tage_allocate(const TageLookup *r, uint8_t outcome)
{
  int start = r->provider + 1;
  int t;

  tageRng = tageRng * 1103515245u + 12345u;
  if (start < TAGE_NUM_TABLES - 1 && ((tageRng >> 16) & 1)) {
    start++;
  }

  for (t = start; t < TAGE_NUM_TABLES; t++) {
    TageEntry *e = &tageTable[t][r->idx[t]];
    if (e->u == 0) {
      e->ctr = (outcome == TAKEN) ? 0 : -1;   // weakly toward the outcome
      e->tag = r->tag[t];
      e->u   = 0;
      return;
    }
  }
  for (t = start; t < TAGE_NUM_TABLES; t++) {
    TageEntry *e = &tageTable[t][r->idx[t]];
    if (e->u > 0) {
      e->u--;
    }
  }
}

static void
tage_train(uint32_t pc, uint8_t outcome)
{
  TageLookup r = tage_lookup(pc);
  int        t;

  // 1. Train the provider (or the bimodal table if nothing matched).
  if (r.provider >= 0) {
    TageEntry *e = &tageTable[r.provider][r.idx[r.provider]];

    // Learn whether a newly allocated provider or the alternate is more
    // trustworthy, from the cases where they disagree.
    if (r.weakNew && r.providerPred != r.altPred) {
      if (r.altPred == outcome) {
        if (tageUseAlt < TAGE_USE_ALT_MAX) tageUseAlt++;
      } else if (tageUseAlt > 0) {
        tageUseAlt--;
      }
    }

    // The provider is useful when it was right and the alternate was not.
    if (r.providerPred != r.altPred) {
      e->u = (r.providerPred == outcome) ? 1 : 0;
    }

    e->ctr = tage_ctr_update(e->ctr, outcome);

    // A likely-new provider has little history yet, so keep training the
    // alternate (or the bimodal table) as well.
    if (r.weakNew) {
      if (r.alt >= 0) {
        TageEntry *a = &tageTable[r.alt][r.idx[r.alt]];
        a->ctr = tage_ctr_update(a->ctr, outcome);
      } else {
        tageBase[r.baseIdx] = counter_update(tageBase[r.baseIdx], outcome);
      }
    }
  } else {
    tageBase[r.baseIdx] = counter_update(tageBase[r.baseIdx], outcome);
  }

  // 2. Allocate after a misprediction, if a longer-history table exists.
  if (r.pred != outcome && r.provider < TAGE_NUM_TABLES - 1) {
    tage_allocate(&r, outcome);
  }

  // 3. Periodically forget usefulness so stale entries can be replaced.
  if (++tageTick >= (1u << TAGE_U_RESET_LOG)) {
    tageTick = 0;
    for (t = 0; t < TAGE_NUM_TABLES; t++) {
      int i;
      for (i = 0; i < (1 << TAGE_TABLE_LOG); i++) {
        tageTable[t][i].u = 0;
      }
    }
  }

  // 4. Update global history, folded histories and path history.
  tage_ghist_push(outcome);
  for (t = 0; t < TAGE_NUM_TABLES; t++) {
    uint8_t outBit = tage_ghist_bit(tageHistLen[t]);
    fold_update(&tageFoldIdx[t],  outcome, outBit);
    fold_update(&tageFoldTag1[t], outcome, outBit);
    fold_update(&tageFoldTag2[t], outcome, outBit);
  }
  tagePathHist = ((tagePathHist << 1) | (pc & 1)) & low_mask(TAGE_PATH_BITS);
}

//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//
void
init_predictor()
{
  switch (bpType) {
    case GSHARE:     gshare_init();     break;
    case TOURNAMENT: tournament_init(); break;
    case CUSTOM:     tage_init();       break;
    default:         break;
  }
}

// Make a prediction for conditional branch instruction at PC 'pc'
// Returning TAKEN indicates a prediction of taken; returning NOTTAKEN
// indicates a prediction of not taken
//
uint8_t
make_prediction(uint32_t pc)
{
  switch (bpType) {
    case STATIC:     return TAKEN;
    case GSHARE:     return gshare_predict(pc);
    case TOURNAMENT: return tournament_predict(pc);
    case CUSTOM:     return tage_predict(pc);
    default:         break;
  }

  // If there is not a compatable bpType then return NOTTAKEN
  return NOTTAKEN;
}

// Train the predictor the last executed branch at PC 'pc' and with
// outcome 'outcome' (true indicates that the branch was taken, false
// indicates that the branch was not taken)
//
void
train_predictor(uint32_t pc, uint8_t outcome)
{
  switch (bpType) {
    case GSHARE:     gshare_train(pc, outcome);     break;
    case TOURNAMENT: tournament_train(pc, outcome); break;
    case CUSTOM:     tage_train(pc, outcome);       break;
    default:         break;
  }
}
