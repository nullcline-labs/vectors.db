//! Generation-based visited set for HNSW graph traversal.
//!
//! Replaces `HashSet<u32>` with O(1) array indexing. Each `clear()` increments
//! a generation counter instead of zeroing the array, making repeated searches fast.

/// Generation-based visited set. Replaces `HashSet<u32>` with O(1) array indexing.
/// Each `clear()` increments a generation counter; `insert()` compares against current generation.
/// Uses u16 generation to reduce full memset frequency to every 65534 clears.
#[derive(Debug)]
pub struct VisitedSet {
    data: Vec<u16>,
    generation: u16,
}

impl VisitedSet {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![0u16; capacity],
            generation: 1,
        }
    }

    /// Reset the set. O(1) amortized — full memset only every 65534 calls.
    pub fn clear(&mut self) {
        if self.generation == u16::MAX {
            self.data.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    /// Ensure capacity covers at least `cap` elements, growing if needed.
    pub fn ensure_capacity(&mut self, cap: usize) {
        if cap > self.data.len() {
            self.data.resize(cap, 0);
        }
    }

    /// Mark `id` as visited. Returns `true` if it was NOT previously visited (i.e. newly inserted).
    #[inline]
    pub fn insert(&mut self, id: u32) -> bool {
        let idx = id as usize;
        if self.data[idx] == self.generation {
            false
        } else {
            self.data[idx] = self.generation;
            true
        }
    }
}

impl Default for VisitedSet {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_clear() {
        let mut vs = VisitedSet::new(100);
        assert!(vs.insert(0));
        assert!(!vs.insert(0)); // already visited
        assert!(vs.insert(50));

        vs.clear();
        assert!(vs.insert(0)); // fresh after clear
        assert!(vs.insert(50));
    }

    #[test]
    fn test_generation_overflow() {
        let mut vs = VisitedSet::new(10);
        // Drive generation to u16::MAX (start at 1, need 65534 clears)
        for _ in 0..65534 {
            vs.clear();
        }
        assert_eq!(vs.generation, u16::MAX);
        vs.insert(5);

        // Next clear triggers memset and resets to 1
        vs.clear();
        assert_eq!(vs.generation, 1);
        assert!(vs.insert(5)); // should be unvisited after memset
    }
}
