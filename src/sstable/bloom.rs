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
