//! 2048 mini-game: pure game logic (the popup draws it). The agent keeps
//! running underneath while you play.

pub const SIZE: usize = 4;
pub const TARGET: u32 = 2048;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dir {
    Up,
    Down,
    Left,
    Right,
}

pub struct Game {
    pub grid: [[u32; SIZE]; SIZE],
    pub score: u32,
    pub won: bool,
    rng: u64,
}

impl Game {
    pub fn new(seed: u64) -> Self {
        let mut g = Self {
            grid: [[0; SIZE]; SIZE],
            score: 0,
            won: false,
            rng: seed | 1,
        };
        g.spawn();
        g.spawn();
        g
    }

    pub fn seeded() -> Self {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        Self::new(t)
    }

    /// xorshift64*: plenty for a toy game, no dependency needed.
    fn next(&mut self) -> u64 {
        let mut x = self.rng;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// No empty cell and no adjacent equal tiles.
    pub fn over(&self) -> bool {
        for r in 0..SIZE {
            for c in 0..SIZE {
                let v = self.grid[r][c];
                if v == 0
                    || (c + 1 < SIZE && v == self.grid[r][c + 1])
                    || (r + 1 < SIZE && v == self.grid[r + 1][c])
                {
                    return false;
                }
            }
        }
        true
    }

    /// Add a 2 (90%) or 4 (10%) to a random empty cell.
    pub fn spawn(&mut self) -> bool {
        let empties: Vec<(usize, usize)> = (0..SIZE)
            .flat_map(|r| (0..SIZE).map(move |c| (r, c)))
            .filter(|&(r, c)| self.grid[r][c] == 0)
            .collect();
        if empties.is_empty() {
            return false;
        }
        let (r, c) = empties[(self.next() % empties.len() as u64) as usize];
        self.grid[r][c] = if self.next().is_multiple_of(10) { 4 } else { 2 };
        true
    }

    #[allow(clippy::needless_range_loop)] // 2-D index math reads clearer
    fn lines(&self, d: Dir) -> [[u32; SIZE]; SIZE] {
        let mut out = [[0; SIZE]; SIZE];
        for i in 0..SIZE {
            for j in 0..SIZE {
                out[i][j] = match d {
                    Dir::Left => self.grid[i][j],
                    Dir::Right => self.grid[i][SIZE - 1 - j],
                    Dir::Up => self.grid[j][i],
                    Dir::Down => self.grid[SIZE - 1 - j][i],
                };
            }
        }
        out
    }

    #[allow(clippy::needless_range_loop)]
    fn write(&mut self, d: Dir, lines: [[u32; SIZE]; SIZE]) {
        for i in 0..SIZE {
            for j in 0..SIZE {
                let v = lines[i][j];
                match d {
                    Dir::Left => self.grid[i][j] = v,
                    Dir::Right => self.grid[i][SIZE - 1 - j] = v,
                    Dir::Up => self.grid[j][i] = v,
                    Dir::Down => self.grid[SIZE - 1 - j][i] = v,
                }
            }
        }
    }

    /// Compress and merge one line towards index 0.
    pub fn merge_left(line: [u32; SIZE]) -> ([u32; SIZE], u32) {
        let tiles: Vec<u32> = line.iter().copied().filter(|v| *v != 0).collect();
        let mut out = [0; SIZE];
        let (mut i, mut k, mut gained) = (0, 0, 0);
        while i < tiles.len() {
            if i + 1 < tiles.len() && tiles[i] == tiles[i + 1] {
                out[k] = tiles[i] * 2;
                gained += tiles[i] * 2;
                i += 2;
            } else {
                out[k] = tiles[i];
                i += 1;
            }
            k += 1;
        }
        (out, gained)
    }

    /// Slide and merge; true when anything changed.
    pub fn slide(&mut self, d: Dir) -> bool {
        let lines = self.lines(d);
        let mut new = lines;
        let mut gained = 0;
        for (i, l) in lines.iter().enumerate() {
            let (m, g) = Self::merge_left(*l);
            new[i] = m;
            gained += g;
        }
        if new == lines {
            return false;
        }
        self.score += gained;
        self.write(d, new);
        if !self.won && self.grid.iter().flatten().any(|v| *v >= TARGET) {
            self.won = true;
        }
        self.spawn();
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merging() {
        assert_eq!(Game::merge_left([2, 2, 2, 2]), ([4, 4, 0, 0], 8));
        assert_eq!(Game::merge_left([0, 2, 0, 2]), ([4, 0, 0, 0], 4));
        assert_eq!(Game::merge_left([2, 4, 8, 16]), ([2, 4, 8, 16], 0));
        let mut g = Game::new(7);
        g.grid = [[2, 2, 0, 0], [0; 4], [0; 4], [0; 4]];
        assert!(g.slide(Dir::Left));
        assert_eq!(g.grid[0][0], 4);
        assert_eq!(g.score, 4);
        g.grid = [[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]];
        assert!(g.over());
    }
}
