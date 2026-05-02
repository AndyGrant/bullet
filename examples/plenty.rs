use bullet_gpu::runtime::cuda::Cuda;
use bullet_lib::LocalSettings;
use bullet_lib::TrainingSchedule;
use bullet_lib::TrainingSteps;
use bullet_lib::game::formats::bulletformat::ChessBoard;
use bullet_lib::game::inputs;
use bullet_lib::game::inputs::Chess768;
use bullet_lib::game::outputs::MaterialCount;
use bullet_lib::lr;
use bullet_lib::lr::LrScheduler;
use bullet_lib::nn::optimiser;
use bullet_lib::trainer::save::SavedFormat;
use bullet_lib::value::ValueTrainer;
use bullet_lib::value::ValueTrainerBuilder;
use bullet_lib::value::loader::ViriBinpackLoader;
use bullet_lib::value::loader::viribinpack::ViriFilter;
use bullet_lib::wdl;
use bullet_lib::wdl::WdlScheduler;
use bullet_trainer::optimiser::adam::AdamW;
use rand::{Rng, rng};
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicUsize, Ordering};
use viriformat::chess::board::Board;
use viriformat::chess::chessmove::Move;
use viriformat::dataformat::Filter;
use viriformat::dataformat::WDL;

macro_rules! init {
    (|$sq:ident, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size];
        while $sq < $size {
            res[$sq] = {$($rest)+};
            $sq += 1;
        }
        res
    }};
}

macro_rules! init_add_assign {
    (|$sq:ident, $init:expr, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size + 1];
        let mut val = $init;
        while $sq < $size {
            res[$sq] = val;
            val += {$($rest)+};
            $sq += 1;
        }

        res[$size] = val;

        res
    }};
}

pub struct Side;
impl Side {
    pub const WHITE: usize = 0;
    pub const BLACK: usize = 1;
}

pub struct Piece;
impl Piece {
    pub const EMPTY: usize = 0;
    pub const PAWN: usize = 2;
    pub const KNIGHT: usize = 3;
    pub const BISHOP: usize = 4;
    pub const ROOK: usize = 5;
    pub const QUEEN: usize = 6;
    pub const KING: usize = 7;
}

pub mod offsets {
    use super::indices;

    pub const PAWN: usize = 0;
    pub const KNIGHT: usize = PAWN + 6 * indices::PAWN[0][64]; // pawn attacks: 6 different pieces: own pawn, own knight, own rook, opp pawn, opp knight, opp rook
    pub const BISHOP: usize = KNIGHT + 10 * indices::KNIGHT[64]; // knight attacks: no kings
    pub const ROOK: usize = BISHOP + 8 * indices::BISHOP[64]; // bishop attacks: queens,kings are skipped
    pub const QUEEN: usize = ROOK + 8 * indices::ROOK[64]; // rook attacks: queens,kings are skipped
    pub const KING: usize = QUEEN + 10 * indices::QUEEN[64]; // queen attacks: no kings
    pub const END: usize = KING + 0 * indices::KING[64]; // king attacks: queen and king skipped
}

pub mod indices {
    use super::attacks;

    pub const PAWN: [[usize; 65]; 2] = [
        init_add_assign!(|sq, 0, 64| attacks::PAWN[0][sq].count_ones() as usize),
        init_add_assign!(|sq, 0, 64| attacks::PAWN[1][sq].count_ones() as usize),
    ];
    pub const KNIGHT: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::KNIGHT[sq].count_ones() as usize); // for every square (including index 64, aka total) stores the start index of threats of that square
    pub const BISHOP: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::BISHOP[sq].count_ones() as usize);
    pub const ROOK: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::ROOK[sq].count_ones() as usize);
    pub const QUEEN: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::QUEEN[sq].count_ones() as usize);
    pub const KING: [usize; 65] = init_add_assign!(|sq, 0, 64| attacks::KING[sq].count_ones() as usize);
}

pub mod attacks {
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = A << 7;

    const DIAGS: [u64; 15] = [
        0x0100_0000_0000_0000,
        0x0201_0000_0000_0000,
        0x0402_0100_0000_0000,
        0x0804_0201_0000_0000,
        0x1008_0402_0100_0000,
        0x2010_0804_0201_0000,
        0x4020_1008_0402_0100,
        0x8040_2010_0804_0201,
        0x0080_4020_1008_0402,
        0x0000_8040_2010_0804,
        0x0000_0080_4020_1008,
        0x0000_0000_8040_2010,
        0x0000_0000_0080_4020,
        0x0000_0000_0000_8040,
        0x0000_0000_0000_0080,
    ];

    pub const PAWN: [[u64; 64]; 2] = [
        init!(|sq, 64| {
            if sq < 8 || sq >= 56 {
                0
            } else {
                let bit: u64 = 1u64 << sq;
                let diag = ((bit & !A) << 7) | ((bit & !H) << 9);
                let push = 1u64 << (sq + 8);
                diag | push
            }
        }),
        init!(|sq, 64| {
            if sq < 8 || sq >= 56 {
                0
            } else {
                let bit: u64 = 1u64 << sq;
                let diag = ((bit & !A) >> 9) | ((bit & !H) >> 7);
                let push = 1u64 << (sq - 8);
                diag | push
            }
        }),
    ];

    pub const KNIGHT: [u64; 64] = init!(|sq, 64| {
        let n = 1 << sq;
        let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
        let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
        (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
    });

    pub const BISHOP: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
    });

    pub const ROOK: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        (0xFF << (rank * 8)) ^ (A << file)
    });

    pub const QUEEN: [u64; 64] = init!(|sq, 64| BISHOP[sq] | ROOK[sq]);

    pub const KING: [u64; 64] = init!(|sq, 64| {
        let mut k = 1 << sq;
        k |= (k << 8) | (k >> 8);
        k |= ((k & !A) >> 1) | ((k & !H) << 1);
        k ^ (1 << sq)
    });
}

pub struct Attacks;
impl Attacks {
    pub fn of_piece<const PC: usize>(from: usize, occ: u64) -> u64 {
        match PC {
            Piece::KNIGHT => Attacks::knight(from),
            Piece::BISHOP => Attacks::bishop(from, occ),
            Piece::ROOK => Attacks::rook(from, occ),
            Piece::QUEEN => Attacks::queen(from, occ),
            Piece::KING => Attacks::king(from),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn pawn(sq: usize, side: usize) -> u64 {
        LOOKUP.pawn[side][sq]
    }

    #[inline]
    pub fn knight(sq: usize) -> u64 {
        LOOKUP.knight[sq]
    }

    #[inline]
    pub fn king(sq: usize) -> u64 {
        LOOKUP.king[sq]
    }

    // hyperbola quintessence
    // this gets automatically vectorised when targeting avx or better
    #[inline]
    pub fn bishop(sq: usize, occ: u64) -> u64 {
        let mask = LOOKUP.bishop[sq];

        let mut diag = occ & mask.diag;
        let mut rev1 = diag.swap_bytes();
        diag = diag.wrapping_sub(mask.bit);
        rev1 = rev1.wrapping_sub(mask.swap);
        diag ^= rev1.swap_bytes();
        diag &= mask.diag;

        let mut anti = occ & mask.anti;
        let mut rev2 = anti.swap_bytes();
        anti = anti.wrapping_sub(mask.bit);
        rev2 = rev2.wrapping_sub(mask.swap);
        anti ^= rev2.swap_bytes();
        anti &= mask.anti;

        diag | anti
    }

    // shifted lookup
    // files and ranks are mapped to 1st rank and looked up by occupancy
    #[inline]
    pub fn rook(sq: usize, occ: u64) -> u64 {
        let flip = ((occ >> (sq & 7)) & File::A).wrapping_mul(DIAG);
        let file_sq = (flip >> 57) & 0x3F;
        let files = LOOKUP.file[sq][file_sq as usize];

        let rank_sq = (occ >> RANK_SHIFT[sq]) & 0x3F;
        let ranks = LOOKUP.rank[sq][rank_sq as usize];

        ranks | files
    }

    #[inline]
    pub fn queen(sq: usize, occ: u64) -> u64 {
        Self::bishop(sq, occ) | Self::rook(sq, occ)
    }

    #[inline]
    pub fn xray_rook(sq: usize, occ: u64, blockers: u64) -> u64 {
        let attacks = Self::rook(sq, occ);
        attacks ^ Self::rook(sq, occ ^ (attacks & blockers))
    }

    #[inline]
    pub fn xray_bishop(sq: usize, occ: u64, blockers: u64) -> u64 {
        let attacks = Self::bishop(sq, occ);
        attacks ^ Self::bishop(sq, occ ^ (attacks & blockers))
    }

    pub const fn white_pawn_setwise(pawns: u64) -> u64 {
        ((pawns & !File::A) << 7) | ((pawns & !File::H) << 9)
    }

    pub const fn black_pawn_setwise(pawns: u64) -> u64 {
        ((pawns & !File::A) >> 9) | ((pawns & !File::H) >> 7)
    }
}

struct File;
impl File {
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = Self::A << 7;
}

const EAST: [u64; 64] = init!(|sq, 64| (0xFF << (sq & 56)) ^ (1 << sq) ^ WEST[sq]);
const WEST: [u64; 64] = init!(|sq, 64| (0xFF << (sq & 56)) & ((1 << sq) - 1));
const DIAG: u64 = DIAGS[7];
const DIAGS: [u64; 15] = [
    0x0100_0000_0000_0000,
    0x0201_0000_0000_0000,
    0x0402_0100_0000_0000,
    0x0804_0201_0000_0000,
    0x1008_0402_0100_0000,
    0x2010_0804_0201_0000,
    0x4020_1008_0402_0100,
    0x8040_2010_0804_0201,
    0x0080_4020_1008_0402,
    0x0000_8040_2010_0804,
    0x0000_0080_4020_1008,
    0x0000_0000_8040_2010,
    0x0000_0000_0080_4020,
    0x0000_0000_0000_8040,
    0x0000_0000_0000_0080,
];

// masks for hyperbola quintessence bishop attacks
#[derive(Clone, Copy)]
struct Mask {
    bit: u64,
    diag: u64,
    anti: u64,
    swap: u64,
}

struct Lookup {
    pawn: [[u64; 64]; 2],
    knight: [u64; 64],
    king: [u64; 64],
    bishop: [Mask; 64],
    rank: [[u64; 64]; 64],
    file: [[u64; 64]; 64],
}

static LOOKUP: Lookup = Lookup { pawn: PAWN, knight: KNIGHT, king: KING, bishop: BISHOP, rank: RANK, file: FILE };

const PAWN: [[u64; 64]; 2] = [
    init!(|sq, 64| (((1 << sq) & !File::A) << 7) | (((1 << sq) & !File::H) << 9)),
    init!(|sq, 64| (((1 << sq) & !File::A) >> 9) | (((1 << sq) & !File::H) >> 7)),
];

const KNIGHT: [u64; 64] = init!(|sq, 64| {
    let n = 1 << sq;
    let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
    let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
});

const KING: [u64; 64] = init!(|sq, 64| {
    let mut k = 1 << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & !File::A) >> 1) | ((k & !File::H) << 1);
    k ^ (1 << sq)
});

const BISHOP: [Mask; 64] = init!(|sq, 64|
    let bit = 1 << sq;
    let file = sq & 7;
    let rank = sq / 8;
    Mask {
        bit,
        diag: bit ^ DIAGS[7 + file - rank],
        anti: bit ^ DIAGS[    file + rank].swap_bytes(),
        swap: bit.swap_bytes()
    }
);

const RANK_SHIFT: [usize; 64] = init!(|sq, 64| sq - (sq & 7) + 1);

const RANK: [[u64; 64]; 64] = init!(|sq, 64| init!(|occ, 64| {
    let file = sq & 7;
    let mask = (occ << 1) as u64;
    let east = ((EAST[file] & mask) | (1 << 63)).trailing_zeros() as usize;
    let west = ((WEST[file] & mask) | 1).leading_zeros() as usize ^ 63;
    (EAST[file] ^ EAST[east] | WEST[file] ^ WEST[west]) << (sq - file)
}));

const FILE: [[u64; 64]; 64] =
    init!(|sq, 64| init!(|occ, 64| (RANK[7 - sq / 8][occ].wrapping_mul(DIAG) & File::H) >> (7 - (sq & 7))));

pub const fn line_through(i: usize, j: usize) -> u64 {
    let sq = 1 << j;

    let rank = i / 8;
    let file = i % 8;

    let files = File::A << file;
    if files & sq > 0 {
        return files;
    }

    let ranks = 0xFF << (8 * rank);
    if ranks & sq > 0 {
        return ranks;
    }

    let diags = DIAGS[7 + file - rank];
    if diags & sq > 0 {
        return diags;
    }

    let antis = DIAGS[file + rank].swap_bytes();
    if antis & sq > 0 {
        return antis;
    }

    0
}

const TOTAL_THREATS: usize = 2 * offsets::END;
const TOTAL: usize = TOTAL_THREATS + 768;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static SQRED: AtomicUsize = AtomicUsize::new(0);
static EVALS: AtomicUsize = AtomicUsize::new(0);
static MAX: AtomicUsize = AtomicUsize::new(0);
const TRACK: bool = false;

pub fn print_feature_stats() {
    let count = COUNT.load(Ordering::Relaxed);
    let sqred = SQRED.load(Ordering::Relaxed);
    let evals = EVALS.load(Ordering::Relaxed);
    let max = MAX.load(Ordering::Relaxed);

    let mean = count as f64 / evals as f64;
    let var = sqred as f64 / evals as f64 - mean.powi(2);
    let pct = 1.96 * var.sqrt();

    println!("Total Evals: {evals}");
    println!("Maximum Active Features: {max}");
    println!("Active Features: {mean:.3} +- {pct:.3} (95%)");
}

pub fn map_piece_threat(
    // maps a threat to a feature index (still need to add half width in case of nstm)
    side: usize,   // color of the attacking piece (POV-relative: 0 = POV's own, 1 = POV's opp)
    piece: usize,  // piece type
    src: usize,    // square of piece type
    dest: usize,   // square of interaction
    target: usize, // piece (including color) on square of interaction
    enemy: bool,   // indicates whether "target" is an enemy piece (threat or protection)
) -> Option<usize> {
    match piece {
        Piece::PAWN => map_pawn_threat(side, src, dest, target, enemy),
        Piece::KNIGHT => map_knight_threat(src, dest, target),
        Piece::BISHOP => map_bishop_threat(src, dest, target),
        Piece::ROOK => map_rook_threat(src, dest, target),
        Piece::QUEEN => map_queen_threat(src, dest, target),
        Piece::KING => map_king_threat(src, dest, target),
        _ => unreachable!(),
    }
}

fn below(src: usize, dest: usize, table: &[u64; 64]) -> usize {
    (table[src] & ((1 << dest) - 1)).count_ones() as usize
}

const fn offset_mapping<const N: usize>(a: [usize; N]) -> [usize; 12] {
    let mut res = [usize::MAX; 12];

    let mut i = 0;
    while i < N {
        res[a[i] - 2] = i; // PieceType - 2 gives the STM index (0..<6)
        res[a[i] + 4] = i + N; // PieceType + 4 gives the OPP index (6..<12)
        i += 1;
    }

    res
}

fn target_is(target: usize, piece: usize) -> bool {
    target % 6 == piece - 2
}

fn map_pawn_threat(side: usize, src: usize, dest: usize, target: usize, enemy: bool) -> Option<usize> {
    // target is still a colored piece
    const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::ROOK]);
    // this MAP call results in the following array:
    // [0, 1, MAX, 2, MAX, MAX, 3, 4, MAX, 5, MAX, MAX]

    // pawn <-> bishop threats are covered by the bishop attacking the pawn, same for queen
    // for pawn <-> pawn threats, we don't cover cases where dest > src to avoid duplicates
    // for pawn <-> pawn threats of the same color, there are never duplicates
    if MAP[target] == usize::MAX || (enemy && dest > src && target_is(target, Piece::PAWN)) {
        None
    } else {
        let idx = indices::PAWN[side][src] + below(src, dest, &attacks::PAWN[side]);
        let threat = offsets::PAWN + MAP[target] * indices::PAWN[side][64] + idx;

        assert!(threat < offsets::KNIGHT, "{threat}");

        Some(threat)
    }
}

fn map_knight_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN]);
    // don't duplicate knight threats, only allow where dest < src
    if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::KNIGHT) {
        None
    } else {
        let idx = indices::KNIGHT[src] + below(src, dest, &attacks::KNIGHT);
        let threat = offsets::KNIGHT + MAP[target] * indices::KNIGHT[64] + idx;

        assert!(threat >= offsets::KNIGHT, "{threat}");
        assert!(threat < offsets::BISHOP, "{threat}");

        Some(threat)
    }
}

fn map_bishop_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK]);
    // queen attacks bishop => bishop attacks queen, so ont used here
    if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::BISHOP) {
        None
    } else {
        let idx = indices::BISHOP[src] + below(src, dest, &attacks::BISHOP);
        let threat = offsets::BISHOP + MAP[target] * indices::BISHOP[64] + idx;

        assert!(threat >= offsets::BISHOP, "{threat}");
        assert!(threat < offsets::ROOK, "{threat}");

        Some(threat)
    }
}

fn map_rook_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK]);
    if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::ROOK) {
        None
    } else {
        let idx = indices::ROOK[src] + below(src, dest, &attacks::ROOK);
        let threat = offsets::ROOK + MAP[target] * indices::ROOK[64] + idx;

        assert!(threat >= offsets::ROOK, "{threat}");
        assert!(threat < offsets::QUEEN, "{threat}");

        Some(threat)
    }
}

fn map_queen_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] = offset_mapping([Piece::PAWN, Piece::KNIGHT, Piece::BISHOP, Piece::ROOK, Piece::QUEEN]);
    if MAP[target] == usize::MAX || dest > src && target_is(target, Piece::QUEEN) {
        None
    } else {
        let idx = indices::QUEEN[src] + below(src, dest, &attacks::QUEEN);
        let threat = offsets::QUEEN + MAP[target] * indices::QUEEN[64] + idx;

        assert!(threat >= offsets::QUEEN, "{threat}");
        assert!(threat < offsets::KING, "{threat}");

        Some(threat)
    }
}

fn map_king_threat(src: usize, dest: usize, target: usize) -> Option<usize> {
    const MAP: [usize; 12] = offset_mapping([]);
    if MAP[target] == usize::MAX {
        None
    } else {
        let idx = indices::KING[src] + below(src, dest, &attacks::KING);
        // piece offset + attacking piece offset + local index (made up of square offset + square-local index)
        let threat = offsets::KING + MAP[target] * indices::KING[64] + idx;

        assert!(threat >= offsets::KING, "{threat}");
        assert!(threat < offsets::END, "{threat}");

        Some(threat)
    }
}

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn flip_horizontal(mut bb: u64) -> u64 {
    const K1: u64 = 0x5555555555555555;
    const K2: u64 = 0x3333333333333333;
    const K4: u64 = 0x0f0f0f0f0f0f0f0f;
    bb = ((bb >> 1) & K1) | ((bb & K1) << 1);
    bb = ((bb >> 2) & K2) | ((bb & K2) << 2);
    ((bb >> 4) & K4) | ((bb & K4) << 4)
}

fn map_features<F: FnMut(usize)>(mut bbs: [u64; 8], mut f: F) {
    // horiontal mirror
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
        }
    };

    let mut pieces = [13; 64];
    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            let pc = 6 * side + piece - 2;
            map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
        }
    }

    let mut count = 0;

    let occ = bbs[0] | bbs[1];

    for side in [Side::WHITE, Side::BLACK] {
        let side_offset = offsets::END * side;
        let opps = bbs[side ^ 1];

        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                let threats = match piece {
                    Piece::PAWN => {
                        let push = if side == Side::WHITE { 1u64 << (sq + 8) } else { 1u64 << (sq - 8) };
                        Attacks::pawn(sq, side) | (push & bbs[Piece::PAWN])
                    }
                    Piece::KNIGHT => Attacks::knight(sq),
                    Piece::BISHOP => Attacks::bishop(sq, occ),
                    Piece::ROOK => Attacks::rook(sq, occ),
                    Piece::QUEEN => Attacks::queen(sq, occ),
                    Piece::KING => Attacks::king(sq),
                    _ => unreachable!(),
                } & occ;

                f(TOTAL_THREATS + [0, 384][side] + 64 * (piece - 2) + sq);
                count += 1;
                map_bb(threats, |dest| {
                    let enemy = (1 << dest) & opps > 0;
                    if let Some(idx) = map_piece_threat(side, piece, sq, dest, pieces[dest], enemy) {
                        f(side_offset + idx);
                        count += 1;
                    }
                });
            });
        }
    }

    if TRACK {
        COUNT.fetch_add(count, Ordering::Relaxed);
        SQRED.fetch_add(count * count, Ordering::Relaxed);
        let evals = EVALS.fetch_add(1, Ordering::Relaxed);
        MAX.fetch_max(count, Ordering::Relaxed);

        if (evals + 1) % (16384 * 6104) == 0 {
            print_feature_stats();
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct ThreatInputs;
impl inputs::SparseInputType for ThreatInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        TOTAL
    }

    fn max_active(&self) -> usize {
        128 + 32
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let mut stm_count = 0;
        let mut stm_feats = [0; 128];
        map_features(bbs, |stm| {
            stm_feats[stm_count] = stm;
            stm_count += 1;
        });

        bbs.swap(0, 1);
        for bb in &mut bbs {
            *bb = bb.swap_bytes();
        }

        let mut ntm_count = 0;
        let mut ntm_feats = [0; 128];
        map_features(bbs, |ntm| {
            ntm_feats[ntm_count] = ntm;
            ntm_count += 1;
        });

        assert_eq!(stm_count, ntm_count);

        for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
            f(stm, ntm);
        }
    }

    fn shorthand(&self) -> String {
        format!("{TOTAL}")
    }

    fn description(&self) -> String {
        "Threat inputs".to_string()
    }
}

fn map_features_bucketed<F: FnMut(usize)>(mut bbs: [u64; 8], mut f: F) {
    // horiontal mirror
    let ksq = (bbs[0] & bbs[Piece::KING]).trailing_zeros();
    if ksq % 8 > 3 {
        for bb in bbs.iter_mut() {
            *bb = flip_horizontal(*bb);
        }
    };

    let mut pieces = [13; 64];
    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            let pc = 6 * side + piece - 2;
            map_bb(bbs[side] & bbs[piece], |sq| pieces[sq] = pc);
        }
    }

    let mut count = 0;

    let occ = bbs[0] | bbs[1];

    for side in [Side::WHITE, Side::BLACK] {
        let side_offset = offsets::END * side;
        let opps = bbs[side ^ 1];

        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                let threats = match piece {
                    Piece::PAWN => {
                        let push = if side == Side::WHITE { 1u64 << (sq + 8) } else { 1u64 << (sq - 8) };
                        Attacks::pawn(sq, side) | (push & bbs[Piece::PAWN])
                    }
                    Piece::KNIGHT => Attacks::knight(sq),
                    Piece::BISHOP => Attacks::bishop(sq, occ),
                    Piece::ROOK => Attacks::rook(sq, occ),
                    Piece::QUEEN => Attacks::queen(sq, occ),
                    Piece::KING => Attacks::king(sq),
                    _ => unreachable!(),
                } & occ;

                count += 1;
                map_bb(threats, |dest| {
                    let enemy = (1 << dest) & opps > 0;
                    if let Some(idx) = map_piece_threat(side, piece, sq, dest, pieces[dest], enemy) {
                        f(side_offset + idx);
                        count += 1;
                    }
                });
            });
        }
    }

    if TRACK {
        COUNT.fetch_add(count, Ordering::Relaxed);
        SQRED.fetch_add(count * count, Ordering::Relaxed);
        let evals = EVALS.fetch_add(1, Ordering::Relaxed);
        MAX.fetch_max(count, Ordering::Relaxed);

        if (evals + 1) % (16384 * 6104) == 0 {
            print_feature_stats();
        }
    }
}

fn get_num_buckets<const N: usize>(arr: &[usize; N]) -> usize {
    let mut max = 0;
    for &val in arr {
        max = max.max(val)
    }
    max + 1
}

#[derive(Clone, Copy, Debug)]
pub struct ThreatInputsBucketsMirrored {
    buckets: [usize; 64],
    num_buckets: usize,
}
impl Default for ThreatInputsBucketsMirrored {
    fn default() -> Self {
        Self { buckets: [0; 64], num_buckets: 1 }
    }
}
impl ThreatInputsBucketsMirrored {
    pub fn new(buckets: [usize; 32]) -> Self {
        let num_buckets = get_num_buckets(&buckets);

        let mut expanded = [0; 64];
        for (idx, elem) in expanded.iter_mut().enumerate() {
            *elem = buckets[(idx / 8) * 4 + [0, 1, 2, 3, 3, 2, 1, 0][idx % 8]];
        }

        Self { buckets: expanded, num_buckets }
    }
}
impl inputs::SparseInputType for ThreatInputsBucketsMirrored {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        768 + TOTAL_THREATS + 768 * self.num_buckets
    }

    fn max_active(&self) -> usize {
        128 + 32 // integrated factoriser
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        let get = |ksq| (if ksq % 8 > 3 { 7 } else { 0 }, 768 * self.buckets[usize::from(ksq)]);
        let (stm_flip, stm_bucket) = get(pos.our_ksq());
        let (ntm_flip, ntm_bucket) = get(pos.opp_ksq());
        Chess768.map_features(pos, |stm, ntm| {
            let bucketed_offset = 768 + TOTAL_THREATS;
            f(bucketed_offset + stm_bucket + (stm ^ stm_flip), bucketed_offset + ntm_bucket + (ntm ^ ntm_flip)); // bucketed feature
            f(stm ^ stm_flip, ntm ^ ntm_flip) // factorised feature
        });

        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let mut stm_count = 0;
        let mut stm_feats = [0; 128];
        map_features_bucketed(bbs, |stm| {
            stm_feats[stm_count] = stm;
            stm_count += 1;
        });

        bbs.swap(0, 1);
        for bb in &mut bbs {
            *bb = bb.swap_bytes();
        }

        let mut ntm_count = 0;
        let mut ntm_feats = [0; 128];
        map_features_bucketed(bbs, |ntm| {
            ntm_feats[ntm_count] = ntm;
            ntm_count += 1;
        });

        assert_eq!(stm_count, ntm_count);

        for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
            f(768 + stm, 768 + ntm); // factoriser offset
        }
    }

    fn shorthand(&self) -> String {
        format!("{TOTAL_THREATS}+{}x{}", Chess768.shorthand(), self.num_buckets)
    }

    fn description(&self) -> String {
        "Threat inputs bucketed mirrored factorised".to_string()
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone)]
struct NetConfig<'a> {
    name: &'a str,
    superbatch: usize,
}

const TRAINING_DIR: &str = "/mnt/d/Chess Data/Selfgen/Training";

fn make_trainer() -> ValueTrainer<AdamW<Cuda>, ThreatInputsBucketsMirrored, MaterialCount<8>> {
    #[rustfmt::skip]
    let inputs = ThreatInputsBucketsMirrored::new([
            00, 01, 02, 03,
            04, 05, 06, 07,
            08, 08, 09, 09,
            10, 10, 10, 10,
            11, 11, 11, 11,
            11, 11, 11, 11,
            11, 11, 11, 11,
            11, 11, 11, 11,
        ]);
    const KING_BUCKETS: usize = 12;
    const OUTPUT_BUCKETS: usize = 8;
    const L1_SIZE: usize = 640;
    const L2_SIZE: usize = 16;
    const L3_SIZE: usize = 32;

    #[rustfmt::skip]
    return ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(optimiser::AdamW)
        .loss_fn(|output, targets| output.sigmoid().squared_error(targets))
        .inputs(inputs)
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w"),
            SavedFormat::id("l0b"),
            SavedFormat::id("l1w"),
            SavedFormat::id("l1b"),
            SavedFormat::id("l2w"),
            SavedFormat::id("l2b"),
            SavedFormat::id("l3w"),
            SavedFormat::id("l3b"),
        ])
        .build(|builder, stm, ntm, buckets| {
            // Build layers
            let l0 = builder.new_affine("l0", 768 + TOTAL_THREATS + 768 * KING_BUCKETS, L1_SIZE);
            let l1 = builder.new_affine("l1", L1_SIZE, OUTPUT_BUCKETS * L2_SIZE);
            let l2 = builder.new_affine("l2", 2 * L2_SIZE, OUTPUT_BUCKETS * L3_SIZE);
            let l3 = builder.new_affine("l3", L3_SIZE + 2 * L2_SIZE, OUTPUT_BUCKETS);

            // Crelu + Pairwise
            let stm_subnet = l0.forward(stm).crelu().pairwise_mul();
            let ntm_subnet = l0.forward(ntm).crelu().pairwise_mul();
            let pairwise_out = stm_subnet.concat(ntm_subnet);
            // Dual activation
            let l1_out = l1.forward(pairwise_out).select(buckets);
            let l1_out = l1_out.concat(l1_out.abs_pow(2.0));
            let l1_out = l1_out.crelu();
            // L2 + L3 forward
            let l2_out = l2.forward(l1_out).select(buckets).screlu();
            let l3_out = l3.forward(l2_out.concat(l1_out)).select(buckets);

            l3_out
        });
}

fn make_settings(experiment_name: &str) -> LocalSettings<'_> {
    return LocalSettings {
        threads: 4,
        output_directory: format!("{}/{}", TRAINING_DIR, experiment_name),
        test_set: None,
        batch_queue_size: 512,
    };
}

fn piece_count_acceptance(board: &Board) -> f64 {
    #[rustfmt::skip]
    const DESIRED_DISTRIBUTION: [f64; 33] = [
        0.018411966423, 0.020641545085, 0.022727271053,
        0.024669162740, 0.026467201733, 0.028121406444,
        0.029631758462, 0.030998276198, 0.032220941240,
        0.033299772000, 0.034234750067, 0.035025893853,
        0.035673184944, 0.036176641754, 0.036536245870,
        0.036752015705, 0.036823932846, 0.036752015705,
        0.036536245870, 0.036176641754, 0.035673184944,
        0.035025893853, 0.034234750067, 0.033299772000,
        0.032220941240, 0.030998276198, 0.029631758462,
        0.028121406444, 0.026467201733, 0.024669162740,
        0.022727271053, 0.020641545085, 0.018411966423,
    ];

    static PIECE_COUNT_STATS: [AtomicU64; 33] = {
        let mut arr: [std::mem::MaybeUninit<AtomicU64>; 33] = [const { MaybeUninit::uninit() }; 33];
        let mut i = 0;
        while i < 33 {
            arr[i].write(AtomicU64::new(0));
            i += 1;
        }
        unsafe { std::mem::transmute::<_, [AtomicU64; 33]>(arr) }
    };
    static PIECE_COUNT_TOTAL: AtomicU64 = AtomicU64::new(0);

    let pc = board.pieces.occupied().count() as usize;
    let count = PIECE_COUNT_STATS[pc].fetch_add(1, Ordering::Relaxed) + 1;
    let total = PIECE_COUNT_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
    let frequency = count as f64 / total as f64;

    // Calculate the acceptance probability for this piece count
    let acceptance = 0.5 * DESIRED_DISTRIBUTION[pc] / frequency;
    acceptance.clamp(0., 1.)
}

fn filter(board: &Board, mv: Move, eval: i16, wdl: f32) -> bool {
    let default_viri_filter = Filter {
        min_ply: 8,
        min_pieces: 0,
        max_eval: 32000,
        filter_tactical: true,
        filter_check: true,
        filter_castling: false,
        max_eval_incorrectness: u32::MAX,
        random_fen_skipping: true,
        random_fen_skip_probability: 0.05,
        wdl_filtered: false,
        wdl_model_params_a: [0.0; 4],
        wdl_model_params_b: [0.0; 4],
        material_min: 17,
        material_max: 78,
        mom_target: 58,
        wdl_heuristic_scale: 1.5,
    };
    let mut rng = rng();
    let wdl = match wdl {
        1.0 => WDL::Win,
        0.5 => WDL::Draw,
        0.0 => WDL::Loss,
        _ => unreachable!(),
    };

    !default_viri_filter.should_filter(mv, eval as i32, board, wdl, &mut rng)
        && rng.random_bool(piece_count_acceptance(board))
}

fn train<WDL: WdlScheduler, LR: LrScheduler>(
    data_path: &str,
    wdl_scheduler: WDL,
    lr_scheduler: LR,
    net: NetConfig,
    load_from: Option<NetConfig>,
) {
    let mut trainer = make_trainer();

    if let Some(load_from_net) = load_from.clone() {
        if load_from_net.name != net.name {
            let _ = trainer
                .optimiser
                .load_weights_from_file(
                    format!(
                        "/mnt/d/Chess Data/Selfgen/Training/{}/net-{}-{}/optimiser_state/weights.bin",
                        load_from_net.name, load_from_net.name, load_from_net.superbatch
                    )
                    .as_str(),
                )
                .unwrap();
        } else {
            let _ = trainer.load_from_checkpoint(
                format!(
                    "/mnt/d/Chess Data/Selfgen/Training/{}/net-{}-{}",
                    load_from_net.name, load_from_net.name, load_from_net.superbatch
                )
                .as_str(),
            );
        }
    }

    let start_superbatch = if load_from.clone().map(|load_from| load_from.name == net.name).unwrap_or(false) {
        load_from.unwrap().superbatch + 1
    } else {
        1
    };

    let schedule = TrainingSchedule {
        net_id: format!("net-{}", net.name).to_string(),
        eval_scale: 450.0,
        steps: TrainingSteps {
            batch_size: 16_384 * 2,
            batches_per_superbatch: 6104 / 2,
            start_superbatch: start_superbatch,
            end_superbatch: net.superbatch,
        },
        wdl_scheduler: wdl_scheduler,
        lr_scheduler: lr_scheduler,
        save_rate: 100,
    };

    trainer.optimiser.set_params(optimiser::AdamWParams {
        decay: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        min_weight: -0.99,
        max_weight: 0.99,
    });

    let settings = make_settings(net.name);
    let data_loader = ViriBinpackLoader::new(data_path, 8192, 12, ViriFilter::Custom(filter));
    trainer.run(&schedule, &settings, &data_loader);
}

fn main() {
    println!("{}", TOTAL_THREATS);
    // Step 1
    train(
        "/mnt/e/Chess/Data/combined.vf",
        wdl::LinearWDL { start: 0.15, end: 0.6 },
        lr::CosineDecayLR { initial_lr: 0.001, final_lr: 0.001 * 0.3 * 0.3 * 0.3, final_superbatch: 1000 },
        NetConfig { name: "0176", superbatch: 1000 },
        None,
    );
}
