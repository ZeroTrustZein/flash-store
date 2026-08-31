use bytes::Bytes;

#[derive(Clone, Debug)]
pub struct BloomFilter {
    bits_per_key: usize,
    k_hashes: u32,
}

impl BloomFilter {
    pub fn new(bits_per_key: usize) -> Self {
        let k = ((bits_per_key as f64) * 0.69314718056).round() as u32; // ln(2)
        let k = k.clamp(1, 30);
        Self {
            bits_per_key,
            k_hashes: k,
        }
    }

    pub fn build_from_keys(&self, keys: &[Bytes]) -> Bytes {
        if keys.is_empty() {
            return Bytes::new();
        }

        let mut bits = keys.len() * self.bits_per_key;
        if bits < 64 {
            bits = 64;
        }
        let bytes = (bits + 7) / 8;
        let mut filter = vec![0u8; bytes + 1]; // last byte stores k
        filter[bytes] = self.k_hashes as u8;

        let num_bits = (bytes * 8) as u32;

        for key in keys {
            let mut h = crc32fast::hash(key);
            let delta = (h >> 17) | (h << 15);
            for _ in 0..self.k_hashes {
                let bit_pos = (h % num_bits) as usize;
                filter[bit_pos / 8] |= 1 << (bit_pos % 8);
                h = h.wrapping_add(delta);
            }
        }

        Bytes::from(filter)
    }

    pub fn may_contain(filter_bytes: &Bytes, key: &[u8]) -> bool {
        if filter_bytes.len() <= 1 {
            return true;
        }

        let bytes_len = filter_bytes.len() - 1;
        let k = filter_bytes[bytes_len];
        let num_bits = (bytes_len * 8) as u32;

        let mut h = crc32fast::hash(key);
        let delta = (h >> 17) | (h << 15);
        for _ in 0..k {
            let bit_pos = (h % num_bits) as usize;
            if (filter_bytes[bit_pos / 8] & (1 << (bit_pos % 8))) == 0 {
                return false;
            }
            h = h.wrapping_add(delta);
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let bloom = BloomFilter::new(10);
        let keys = vec![
            Bytes::from_static(b"apple"),
            Bytes::from_static(b"banana"),
            Bytes::from_static(b"cherry"),
        ];

        let filter = bloom.build_from_keys(&keys);

        // Positive checks (must be true)
        assert!(BloomFilter::may_contain(&filter, b"apple"));
        assert!(BloomFilter::may_contain(&filter, b"banana"));
        assert!(BloomFilter::may_contain(&filter, b"cherry"));

        // Negative check (likely false for non-member)
        let _ = BloomFilter::may_contain(&filter, b"durian");
    }

    #[test]
    fn test_bloom_filter_empty() {
        let bloom = BloomFilter::new(10);
        let filter = bloom.build_from_keys(&[]);
        assert_eq!(filter.len(), 0);
        assert!(BloomFilter::may_contain(&filter, b"any_key"));
    }
}
