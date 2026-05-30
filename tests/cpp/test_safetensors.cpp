// SPDX-License-Identifier: Apache-2.0
#include "engine/model_config.hpp"
#include "engine/safetensors.hpp"

#include <cstdint>
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Prepend the 8-byte little-endian header length to a header + payload.
std::vector<uint8_t> st_buf(const std::string& header, const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> buf;
    const uint64_t len = header.size();
    for (int i = 0; i < 8; ++i) {
        buf.push_back(static_cast<uint8_t>((len >> (8 * i)) & 0xFFu));
    }
    buf.insert(buf.end(), header.begin(), header.end());
    buf.insert(buf.end(), payload.begin(), payload.end());
    return buf;
}

// A minimal valid buffer: F32 [2,2] = {1,2,3,4}, BF16 [2] = {1,2}.
std::vector<uint8_t> build_safetensors() {
    const std::string header = R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},)"
                               R"("b":{"dtype":"BF16","shape":[2],"data_offsets":[16,20]}})";
    std::vector<uint8_t> payload;
    const float f32[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto* fp = reinterpret_cast<const uint8_t*>(f32);
    payload.insert(payload.end(), fp, fp + 16);
    // bf16 little-endian: 1.0 -> 0x3F80 -> {0x80,0x3F}; 2.0 -> 0x4000 -> {0x00,0x40}
    payload.insert(payload.end(), {0x80, 0x3F, 0x00, 0x40});
    return st_buf(header, payload);
}

} // namespace

TEST(SafeTensors, ParsesSyntheticBuffer) {
    const auto buf = build_safetensors();
    const auto st = engine::SafeTensors::parse(buf.data(), buf.size());

    EXPECT_EQ(st.size(), 2u);
    ASSERT_TRUE(st.contains("a"));
    const auto& a = st.get("a");
    EXPECT_EQ(a.numel(), 4);
    EXPECT_FLOAT_EQ(a.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(a.data()[3], 4.0f);

    const auto& b = st.get("b");
    EXPECT_FLOAT_EQ(b.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(b.data()[1], 2.0f);
}

TEST(SafeTensors, MissingTensorThrows) {
    const auto buf = build_safetensors();
    const auto st = engine::SafeTensors::parse(buf.data(), buf.size());
    EXPECT_THROW((void)st.get("absent"), std::runtime_error);
}

TEST(SafeTensorsHardening, RejectsOverflowingHeaderLength) {
    std::vector<uint8_t> buf(16, 0xFF); // header_len = 2^64-1
    EXPECT_THROW((void)engine::SafeTensors::parse(buf.data(), buf.size()), std::runtime_error);
}

TEST(SafeTensorsHardening, RejectsTruncatedHeader) {
    std::vector<uint8_t> buf(8, 0);
    buf[0] = 0xFF; // header_len = 255, but no header bytes follow
    EXPECT_THROW((void)engine::SafeTensors::parse(buf.data(), buf.size()), std::runtime_error);
}

TEST(SafeTensorsHardening, RejectsNegativeDimension) {
    const auto buf = st_buf(R"({"a":{"dtype":"F32","shape":[-1,4],"data_offsets":[0,16]}})",
                            std::vector<uint8_t>(16, 0));
    EXPECT_THROW((void)engine::SafeTensors::parse(buf.data(), buf.size()), std::runtime_error);
}

TEST(SafeTensorsHardening, RejectsByteCountMismatch) {
    const auto buf = st_buf(R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,8]}})",
                            std::vector<uint8_t>(8, 0));
    EXPECT_THROW((void)engine::SafeTensors::parse(buf.data(), buf.size()), std::runtime_error);
}

TEST(SafeTensorsHardening, RejectsOverlappingRegions) {
    const auto buf = st_buf(R"({"a":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},)"
                            R"("b":{"dtype":"F32","shape":[2,2],"data_offsets":[8,24]}})",
                            std::vector<uint8_t>(24, 0));
    EXPECT_THROW((void)engine::SafeTensors::parse(buf.data(), buf.size()), std::runtime_error);
}

TEST(SafeTensorsHardening, NormalizesMalformedHeaderToRuntimeError) {
    // Missing data_offsets would raise nlohmann::json::out_of_range; must surface as runtime_error.
    const auto buf = st_buf(R"({"a":{"dtype":"F32","shape":[2,2]}})", std::vector<uint8_t>(16, 0));
    EXPECT_THROW((void)engine::SafeTensors::parse(buf.data(), buf.size()), std::runtime_error);
}

TEST(SafeTensorsHardening, RejectsUnsupportedDtype) {
    const auto buf = st_buf(R"({"a":{"dtype":"I64","shape":[2],"data_offsets":[0,16]}})",
                            std::vector<uint8_t>(16, 0));
    EXPECT_THROW((void)engine::SafeTensors::parse(buf.data(), buf.size()), std::runtime_error);
}

TEST(ModelConfig, ParsesQwen2FieldsAndDerivesHeadDim) {
    const std::string cfg = R"({
        "model_type": "qwen2", "hidden_size": 896, "num_hidden_layers": 24,
        "num_attention_heads": 14, "num_key_value_heads": 2, "intermediate_size": 4864,
        "vocab_size": 151936, "max_position_embeddings": 32768, "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-6, "tie_word_embeddings": true, "hidden_act": "silu"
    })";
    const auto c = engine::ModelConfig::from_json(cfg);
    EXPECT_EQ(c.hidden_size, 896);
    EXPECT_EQ(c.num_hidden_layers, 24);
    EXPECT_EQ(c.num_key_value_heads, 2);
    EXPECT_EQ(c.head_dim, 64); // derived from 896 / 14
    EXPECT_TRUE(c.tie_word_embeddings);
    EXPECT_TRUE(c.attention_qkv_bias); // qwen2 -> bias on q/k/v
}

TEST(ModelConfig, RejectsZeroAttentionHeads) {
    const std::string cfg = R"({
        "model_type": "qwen2", "hidden_size": 896, "num_hidden_layers": 24,
        "num_attention_heads": 0, "intermediate_size": 4864, "vocab_size": 151936
    })";
    EXPECT_THROW((void)engine::ModelConfig::from_json(cfg), std::runtime_error);
}

TEST(ModelConfig, NonQwen2UsesAttentionBiasAndExplicitHeadDim) {
    const std::string cfg = R"({
        "model_type": "llama", "hidden_size": 2048, "num_hidden_layers": 16,
        "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 64,
        "intermediate_size": 8192, "vocab_size": 128256, "attention_bias": false,
        "rms_norm_eps": 1e-5
    })";
    const auto c = engine::ModelConfig::from_json(cfg);
    EXPECT_EQ(c.head_dim, 64); // explicit, used as-is
    EXPECT_EQ(c.num_key_value_heads, 8);
    EXPECT_FALSE(c.attention_qkv_bias); // non-qwen2 follows attention_bias
}

TEST(SafeTensorsReal, LoadsQwenWeightsIfPresent) {
    const std::filesystem::path path = std::filesystem::path(ENGINE_SOURCE_DIR) / "weights" /
                                       "Qwen2.5-0.5B-Instruct" / "model.safetensors";
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "weights absent; run scripts/fetch_model.py to enable";
    }
    const auto st = engine::SafeTensors::load(path.string());
    ASSERT_TRUE(st.contains("model.embed_tokens.weight"));
    const auto& emb = st.get("model.embed_tokens.weight");
    ASSERT_EQ(emb.ndim(), 2u);
    EXPECT_EQ(emb.dim(0), 151936);
    EXPECT_EQ(emb.dim(1), 896);
    EXPECT_TRUE(st.contains("model.layers.0.self_attn.q_proj.bias")); // qwen2 qkv bias
    EXPECT_FALSE(st.contains("lm_head.weight"));                      // tied embeddings
}
