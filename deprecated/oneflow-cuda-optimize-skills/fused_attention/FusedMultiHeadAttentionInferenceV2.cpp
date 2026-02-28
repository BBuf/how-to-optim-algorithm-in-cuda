Maybe<void> ParseDims(const std::string& name, const Shape& shape, const std::string& layout,
                      const Optional<int64_t>& batch_size, const Optional<int64_t>& seq_len,
                      const Optional<int64_t>& num_heads, const Optional<int64_t>& head_size,
                      int64_t* b, int64_t* m, int64_t* h, int64_t* k, bool* bm_packed) {
  if (shape.NumAxes() == 2) {
    if (layout == "(BM)(HK)" || layout == "(BM)(H2K)" || layout == "(BM)(H3K)") {
      *bm_packed = true;
      CHECK_OR_RETURN(batch_size);
      CHECK_OR_RETURN(seq_len);
      *b = JUST(batch_size);
      *m = JUST(seq_len);
      int64_t packed_n = 0;
      if (layout == "(BM)(HK)") {
        packed_n = 1;
      } else if (layout == "(BM)(H2K)") {
        CHECK_NE_OR_RETURN(name, "query") << "query_layout should not be '(BM)(H2K)'";
        packed_n = 2;
      } else if (layout == "(BM)(H3K)") {
        packed_n = 3;
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
      const int64_t hidden_size = shape.At(1);
      if (num_heads) {
        const int64_t expected_h = JUST(num_heads);
        const int64_t packed_h = packed_n * expected_h;
        CHECK_EQ_OR_RETURN(hidden_size % packed_h, 0)
            << "The size of the last dimension of the " << name
            << " tensor should be a multiple of " << packed_h << ".";
        *h = expected_h;
        *k = hidden_size / packed_h;
      } else if (head_size) {
        const int64_t expected_k = JUST(head_size);
        const int64_t packed_k = expected_k * packed_n;
        CHECK_EQ_OR_RETURN(hidden_size % packed_k, 0)
            << "The size of the last dimension of the " << name
            << " tensor should be a multiple of " << packed_k << ".";
        *h = hidden_size / packed_k;
        *k = expected_k;
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN() << name
                                  << "_layout should be '(BM)(HK)', '(BM)(H2K)', or '(BM)(H3K)' "
                                     "when the number of dimensions of "
                                  << name << " tensor is 2.";
    }
  } else if (shape.NumAxes() == 3) {
    if (layout == "BM(HK)" || layout == "MB(HK)" || layout == "BM(H2K)" || layout == "MB(H2K)"
        || layout == "BM(H3K)" || layout == "MB(H3K)") {
      *bm_packed = false;
      int64_t packed_n = 0;
      if (layout == "BM(HK)") {
        *b = shape.At(0);
        *m = shape.At(1);
        packed_n = 1;
      } else if (layout == "MB(HK)") {
        *b = shape.At(1);
        *m = shape.At(0);
        packed_n = 1;
      } else if (layout == "BM(H2K)") {
        CHECK_NE_OR_RETURN(name, "query") << "query_layout should not be 'BM(H2K)'";
        *b = shape.At(0);
        *m = shape.At(1);
        packed_n = 2;
      } else if (layout == "MB(H2K)") {
        CHECK_NE_OR_RETURN(name, "query") << "query_layout should not be 'MB(H2K)'";
        *b = shape.At(1);
        *m = shape.At(0);
        packed_n = 2;
      } else if (layout == "BM(H3K)") {
        *b = shape.At(0);
        *m = shape.At(1);
        packed_n = 3;
      } else if (layout == "MB(H3K)") {
        *b = shape.At(1);
        *m = shape.At(0);
        packed_n = 3;
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
      const int64_t hidden_size = shape.At(2);
      if (num_heads) {
        const int64_t expected_h = JUST(num_heads);
        const int64_t packed_h = packed_n * expected_h;
        CHECK_EQ_OR_RETURN(hidden_size % packed_h, 0)
            << "The size of the last dimension of the " << name
            << " tensor should be a multiple of " << packed_h << ".";
        *h = expected_h;
        *k = hidden_size / packed_h;
      } else if (head_size) {
        const int64_t expected_k = JUST(head_size);
        const int64_t packed_k = expected_k * packed_n;
        CHECK_EQ_OR_RETURN(hidden_size % packed_k, 0)
            << "The size of the last dimension of the " << name
            << " tensor should be a multiple of " << packed_k << ".";
        *h = hidden_size / packed_k;
        *k = expected_k;
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    } else if (layout == "(BM)HK") {
      *bm_packed = true;
      CHECK_OR_RETURN(batch_size);
      CHECK_OR_RETURN(seq_len);
      *b = JUST(batch_size);
      *m = JUST(seq_len);
      *h = shape.At(1);
      *k = shape.At(2);
    } else {
      UNIMPLEMENTED_THEN_RETURN()
          << name
          << "_layout should be 'BM(HK)', 'MB(HK)', 'BM(H2K)', 'MB(H2K)', 'BM(H3K)', "
             "'MB(H3K)' or '(BM)HK' when the number of dimensions of "
          << name << " tensor is 3.";
    }
  } else if (shape.NumAxes() == 4) {
    *bm_packed = false;
    if (layout == "BMHK") {
      *b = shape.At(0);
      *m = shape.At(1);
      *h = shape.At(2);
      *k = shape.At(3);
    } else if (layout == "BHMK") {
      *b = shape.At(0);
      *m = shape.At(2);
      *h = shape.At(1);
      *k = shape.At(3);
    } else if (layout == "MBHK") {
      *b = shape.At(1);
      *m = shape.At(0);
      *h = shape.At(2);
      *k = shape.At(3);
    } else {
      UNIMPLEMENTED_THEN_RETURN()
          << name << "_layout should be 'BMHK', 'BHMK' or 'MBHK' when the number of dimensions of "
          << name << " tensor is 4.";
    }
  } else {
    UNIMPLEMENTED_THEN_RETURN() << "The number of dimensions of the " << name
                                << " tensor should be 3 or 4";
  };
  if (batch_size) {
    const int64_t expected_b = JUST(batch_size);
    CHECK_EQ_OR_RETURN(*b, expected_b)
        << "The size of dimension 'B' of " << name << " tensor should be " << expected_b << ".";
  }
  if (seq_len) {
    const int64_t expected_m = JUST(seq_len);
    CHECK_EQ_OR_RETURN(*m, expected_m)
        << "The size of dimension 'M' of " << name << " tensor should be " << expected_m << ".";
  }
  if (num_heads) {
    const int64_t expected_h = JUST(num_heads);
    CHECK_EQ_OR_RETURN(*h, expected_h)
        << "The size of dimension 'H' of " << name << " tensor should be " << expected_h << ".";
  }
  if (head_size) {
    const int64_t expected_k = JUST(head_size);
    CHECK_EQ_OR_RETURN(*k, expected_k)
        << "The size of dimension 'K' of " << name << " tensor should be " << expected_k << ".";
  }
  return Maybe<void>::Ok();
}

Maybe<void> ParseDims(const std::string& name, const Shape& shape, const std::string& layout,
                      const Optional<int64_t>& num_heads, const Optional<int64_t>& head_size,
                      int64_t* b, int64_t* m, int64_t* h, int64_t* k) {
  bool bm_packed{};
  return ParseDims(name, shape, layout, Optional<int64_t>(), Optional<int64_t>(), num_heads,
                   head_size, b, m, h, k, &bm_packed);
}

}  // namespace

class FusedMultiHeadAttentionInferenceFunctor {
 public:
  FusedMultiHeadAttentionInferenceFunctor() = default;
  Maybe<Tensor> operator()(
      const std::shared_ptr<one::Tensor>& query, const std::shared_ptr<one::Tensor>& key,
      const std::shared_ptr<one::Tensor>& value, const int64_t& num_heads, const bool& causal,
      const int64_t& query_hidden_slice_start, const int64_t& query_hidden_slice_end,
      const int64_t& key_hidden_slice_start, const int64_t& key_hidden_slice_end,
      const int64_t& value_hidden_slice_start, const int64_t& value_hidden_slice_end,
      const Optional<one::Tensor>& attn_bias, const int64_t& causal_diagonal_offset) const {
    CHECK_OR_RETURN(query_hidden_slice_start == 0 && key_hidden_slice_start == 0
                    && value_hidden_slice_start == 0 && query_hidden_slice_end == -1
                    && key_hidden_slice_end == -1 && value_hidden_slice_end == -1)
        << "The parameters 'query_hidden_slice_start', 'query_hidden_slice_end', "
           "'key_hidden_slice_start', 'key_hidden_slice_end', 'value_hidden_slice_start', "
           "'value_hidden_slice_end' have been deprecated.";

    const int64_t query_hidden_size = query->shape()->At(2);
    CHECK_EQ_OR_RETURN(query_hidden_size % num_heads, 0)
        << "The hidden size of the query tensor should be a multiple of num_heads.";
    const int64_t query_head_size = query_hidden_size / num_heads;
    return functional::FusedMultiHeadAttentionInferenceV2(
        query, "BM(HK)", query_head_size, Optional<one::Tensor>(), Optional<int64_t>(), key,
        "BM(HK)", Optional<one::Tensor>(), Optional<one::Tensor>(), Optional<int64_t>(), value,
        "BM(HK)", attn_bias, "BM(HK)", Optional<float>(), causal, Optional<std::string>(),
        causal_diagonal_offset);
  }
};

class FusedMultiHeadAttentionInferenceV2Functor {
 public:
  // 定义了一个 OpExprCacheKey 结构体,用于缓存 OpExpr (运算表达式)。
  // 它包含 3 个布尔类型字段,表示是否有 attn_bias, seq_start ,key_seq_len 输入。
  // 还定义了==和Hash运算符,用于后续的unordered_map中查找OpExpr。
  struct OpExprCacheKey {
    bool has_attn_bias = false;
    bool has_seq_start = false;
    bool has_key_seq_len = false;
    bool operator==(const OpExprCacheKey& rhs) const {
      return this->has_attn_bias == rhs.has_attn_bias && this->has_seq_start == rhs.has_seq_start
             && this->has_key_seq_len == rhs.has_key_seq_len;
    }
  };
  struct OpExprCacheKeyHash {
    size_t operator()(const OpExprCacheKey& key) const {
      return Hash(key.has_attn_bias, key.has_seq_start, key.has_key_seq_len);
    }
  };
  // 定义了一个 unordered_map, 键值为 OpExprCacheKey, 值类型为 std::shared_ptr<OpExpr>, 用于缓存运算表达式。
  using OpExprCache =
      std::unordered_map<OpExprCacheKey, std::shared_ptr<OpExpr>, OpExprCacheKeyHash>;
  // 在构造函数中,遍历 attn_bias,seq_start 和 key_seq_len 的所有组合,构建对应的 multi_head_attention_inference 运算表达式,并缓存到 op_cache_ 中。
  // 后续可以通过 OpExprCacheKey 快速从 op_cache_ 中查找对应的 OpExp r,从而加速推理进程。
  FusedMultiHeadAttentionInferenceV2Functor() {
    for (bool has_attn_bias : {false, true}) {
      for (bool has_seq_start : {false, true}) {
        for (bool has_key_seq_len : {false, true}) {
          auto builder = one::OpBuilder("fused_multi_head_attention_inference")
                             .Input("query")
                             .Input("key")
                             .Input("value");
          if (has_attn_bias) { builder.Input("attn_bias"); }
          if (has_seq_start) { builder.Input("query_seq_start").Input("key_seq_start"); }
          if (has_key_seq_len) { builder.Input("key_seq_len"); }
          auto op = CHECK_JUST(builder.Output("out").Build());
          OpExprCacheKey key;
          key.has_attn_bias = has_attn_bias;
          key.has_seq_start = has_seq_start;
          key.has_key_seq_len = has_key_seq_len;
          op_cache_.emplace(key, op);
        }
      }
    }
  }
  Maybe<Tensor> operator()(
      const std::shared_ptr<one::Tensor>& query, const std::string& query_layout,
      const Optional<int64_t>& query_head_size, const Optional<one::Tensor>& query_seq_start,
      const Optional<int64_t>& query_max_seq_len, const Optional<one::Tensor>& key,
      const Optional<std::string>& key_layout, const Optional<one::Tensor>& key_seq_start,
      const Optional<one::Tensor>& key_seq_len, const Optional<int64_t>& key_max_seq_len,
      const Optional<one::Tensor>& value, const Optional<std::string>& value_layout,
      const Optional<one::Tensor>& attn_bias, const std::string& output_layout,
      const Optional<float>& scale, const Optional<bool>& causal,
      const Optional<std::string>& attn_mask_type, const int64_t& causal_diagonal_offset) const {
    // 确定 mask 这个对角矩阵长什么样，比如 causal_from_top_left 表示这个对角线矩阵是左上角的
    std::string attn_mask_type_val = "none";
    if (attn_mask_type) {
      // 确保 causal 为空,因为 attn_mask_type 和 causal 不能同时指定
      CHECK(!causal) << "Only one of attn_mask_type and causal can be specified at the same time.";
      // 将 attn_mask_type_val 设置为 attn_mask_type 的值
      attn_mask_type_val = *JUST(attn_mask_type);
      // 检查 attn_mask_type_val 的值必须为 "none"、"causal_from_top_left" 或 "causal_from_bottom_right" 中的一个
      CHECK_OR_RETURN(attn_mask_type_val == "none" || attn_mask_type_val == "causal_from_top_left"
                      || attn_mask_type_val == "causal_from_bottom_right")
          << "The value of attn_mask_type should be one of 'none', 'causal_from_top_left' or "
             "'causal_from_bottom_right'";
    } else if (causal && JUST(causal)) {
      // 否则,如果 causal 有值,则将 attn_mask_type_val 设置为 "causal_from_top_left"
      attn_mask_type_val = "causal_from_top_left";
    } else {
      // do nothing
    }
    // causal_diagonal_offset 表示对角线矩阵的起始位置
    CHECK_GE_OR_RETURN(causal_diagonal_offset, 0)
        << "The value of causal_diagonal_offset should be greater or equal to 0.";

    // 定义一个可选的 batch size 变量。后续会根据是否有 seq_start 输入来决定是否对其赋值。
    Optional<int64_t> batch_size;
    std::shared_ptr<one::Tensor> query_seq_start_tensor;
    std::shared_ptr<one::Tensor> key_seq_start_tensor;
    // 1. 如果 query_seq_start 有值,则执行如下检查:
    //   - key_seq_start 也必须有值,两个变量要同时为空或同时不为空
    //   - query_max_seq_len 和 key_max_seq_len 不能为空
    //   - 将 query_seq_start 和 key_seq_start 赋值给之前定义的张量变量
    //   - 检查 query_seq_start_tensor 的维度必须为1
    //   - 检查 query_seq_start_tensor 和 key_seq_start_tensor 的形状必须相同
    //   - 检查 query_seq_start_tensor 的大小必须大于1
    //   - 从 query_seq_start_tensor 获取 batch_size
    //   - 如果 key_seq_len 有值,检查它的维度为1,大小等于 batch_size
    // 2. 否则,如果 query_seq_start 为空,则
    //   - key_seq_start 也必须为空
    //   - key_seq_len 也必须为空
    if (query_seq_start) {
      CHECK_OR_RETURN(key_seq_start) << "The tensors query_seq_start and key_seq_start should both "
                                        "be None or both not be None at the same time.";
      CHECK_OR_RETURN(query_max_seq_len)
          << "query_max_seq_len should not be None when query_seq_start is not None.";
      CHECK_OR_RETURN(key_max_seq_len)
          << "key_max_seq_len should not be None when key_seq_start is not None.";
      query_seq_start_tensor = JUST(query_seq_start);
      key_seq_start_tensor = JUST(key_seq_start);
      CHECK_EQ_OR_RETURN(query_seq_start_tensor->shape()->NumAxes(), 1)
          << "The number of dimensions of query_seq_start tensor should be 1.";
      CHECK_OR_RETURN(*query_seq_start_tensor->shape() == *key_seq_start_tensor->shape())
          << "The shapes of the query_seq_start and key_seq_start tensors should match.";
      CHECK_GT_OR_RETURN(query_seq_start_tensor->shape()->At(0), 1)
          << "The size of query_seq_start should be greater than 1.";
      // 获取batch_size
      batch_size = query_seq_start_tensor->shape()->At(0) - 1;
      if (key_seq_len) {
        CHECK_EQ_OR_RETURN(JUST(key_seq_len)->shape()->NumAxes(), 1)
            << "The number of dimensions of key_seq_len tensor should be 1.";
        CHECK_EQ_OR_RETURN(JUST(key_seq_len)->shape()->At(0), JUST(batch_size))
            << "The size of the key_seq_len tensor should be " << JUST(batch_size) << ".";
      }
    } else {
      CHECK_OR_RETURN(!key_seq_start)
          << "The tensors query_seq_start and key_seq_start should both "
             "be None or both not be None at the same time.";
      CHECK_OR_RETURN(!key_seq_len)
          << "The key_seq_len tensor should be None when query_seq_start is None.";
    }
    std::shared_ptr<one::Tensor> key_tensor;
    std::string key_tensor_layout;
    std::shared_ptr<one::Tensor> value_tensor;
    std::string value_tensor_layout;

    // query 的 batch_size，seq_length，num_heads，hidden_size_per_attention_head
    int64_t q_b = 0;
    int64_t q_m = 0;
    int64_t q_h = 0;
    int64_t q_k = 0;
    // query 的 batch_size 维度和 seq_length 维度是否 pack 在一起
    bool q_bm_packed = false;
    // 使用 ParseDims 根据 query 的 shape 以及 query_layout 来解析维度
    JUST(ParseDims("query", *query->shape(), query_layout, batch_size, query_max_seq_len,
                   Optional<int64_t>(), query_head_size, &q_b, &q_m, &q_h, &q_k, &q_bm_packed));
    // query 的 hidden_size_per_attention_head 需要满足被 8 整除
    CHECK_EQ_OR_RETURN(q_k % 8, 0)
        << "The size of dimension 'K' of the query tensor should be a multiple of 8.";
    if (q_bm_packed) {
      CHECK_OR_RETURN(query_seq_start)
          << "The query_seq_start tensor should not be None when the query tensor is BM-Packed.";
    }

    // key 的 batch_size，seq_length，num_heads，hidden_size_per_attention_head
    int64_t k_b = 0;
    int64_t k_m = 0;
    int64_t k_h = 0;
    int64_t k_k = 0;
    // key 的 batch_size 维度和 seq_length 维度是否 pack 在一起
    bool k_bm_packed = false;
    if (key) {
      key_tensor = JUST(key);
      key_tensor_layout = *JUST(key_layout);
      // 解析 key_tensor 的维度,得到 k_b、k_m、k_h、k_k 和 k_bm_packed 等值
      JUST(ParseDims("key", *key_tensor->shape(), key_tensor_layout, q_b, key_max_seq_len,
                     Optional<int64_t>(), q_k, &k_b, &k_m, &k_h, &k_k, &k_bm_packed));
      // 检查 k_b 与 q_b 相等, k_h 与 q_h 相等, k_bm_packed 与 q_bm_packed 相等
      CHECK_EQ_OR_RETURN(k_b, q_b) << "The size of dimension 'B' of the key tensor should be the "
                                      "same as that of the query tensor.";
      CHECK_EQ_OR_RETURN(k_h, q_h) << "The size of dimension 'H' of the key tensor should be the "
                                      "same as that of the query tensor.";
      CHECK_EQ_OR_RETURN(k_bm_packed, q_bm_packed)
          << "The query tensor and the key tensor should either both be BM-Packed or both not be "
             "BM-Packed at the same time.";

    } else {
      // 如果 key 为空，则检查 query_layout 必须为 "BM(H3K)" 或 "MB(H3K)"
      CHECK_OR_RETURN(query_layout == "BM(H3K)" || query_layout == "MB(H3K)")
          << "The value of query_layout should be 'BM(H3K)' or 'MB(H3K)' when the key tensor is "
             "None.";
      // 将 query 赋值给 key_tensor
      key_tensor = query;
      // 将 query_layout 赋值给 key_tensor_layout
      key_tensor_layout = query_layout;
      // 将 q_b、q_m、q_h、q_k 和 q_bm_packed 的值赋给 k_b、k_m、k_h、k_k 和 k_bm_packed
      k_b = q_b;
      k_m = q_m;
      k_h = q_h;
      k_k = q_k;
      k_bm_packed = q_bm_packed;
    }

    // value 的 batch_size，seq_length，num_heads，hidden_size_per_attention_head
    int64_t v_b = 0;
    int64_t v_m = 0;
    int64_t v_h = 0;
    int64_t v_k = 0;
    // value 的 batch_size 维度和 seq_length 维度是否 pack 在一起
    bool v_bm_packed = false;
    if (value) {
      value_tensor = JUST(value);
      value_tensor_layout = *JUST(value_layout);
      // 解析 value_tensor 的维度,得到 k_b、k_m、k_h、k_k 和 k_bm_packed 等值
      JUST(ParseDims("value", *value_tensor->shape(), value_tensor_layout, q_b, k_m, q_h,
                     Optional<int64_t>(), &v_b, &v_m, &v_h, &v_k, &v_bm_packed));
      CHECK_EQ_OR_RETURN(v_b, q_b) << "The size of dimension 'B' of the value tensor should be the "
                                      "same as that of the query tensor.";
      CHECK_EQ_OR_RETURN(v_m, k_m) << "The size of dimension 'M' of the value tensor should be the "
                                      "same as that of the key tensor.";
      CHECK_EQ_OR_RETURN(v_k % 8, 0)
          << "The size of dimension 'K' of the value tensor should be a multiple of 8.";
      CHECK_EQ_OR_RETURN(v_bm_packed, k_bm_packed)
          << "The key tensor and the value tensor should either both be BM-Packed or both not be "
             "BM-Packed at the same time.";

    } else {
      CHECK_OR_RETURN(key_tensor_layout == "BM(H2K)" || key_tensor_layout == "MB(H2K)"
                      || key_tensor_layout == "BM(H3K)" || key_tensor_layout == "MB(H3K)")
          << "The value of key_layout should be 'BM(H3K)', 'MB(H3K)', 'BM(H2K)' or 'MB(H2K)' when "
             "the value tensor is None.";
      value_tensor = key_tensor;
      value_tensor_layout = key_tensor_layout;
      v_b = k_b;
      v_m = k_m;
      v_h = k_h;
      v_k = k_k;
      v_bm_packed = k_bm_packed;
    }

    if (attn_bias) {
      // 获取 attn_bias 的形状 attn_bias_shape 和维度数 num_attn_bias_axes
      const auto attn_bias_shape = JUST(attn_bias)->shape();
      const int64_t num_attn_bias_axes = attn_bias_shape->NumAxes();
      // 检查 num_attn_bias_axes 必须大于 0 小于等于 4
      CHECK_OR_RETURN(num_attn_bias_axes > 0 && num_attn_bias_axes <= 4)
          << "The number of dimensions of attn_bias should be greater than 0 and less than or "
             "equal to 4.";
      // 检查 attn_bias_shape 的最后一维(即 -1 维)必须大于等于 key_tensor 的 m 维
      CHECK_GE_OR_RETURN(attn_bias_shape->At(num_attn_bias_axes - 1), k_m)
          << "The size of the -1 dimension of attn_bias should be greater than or equal to the "
             "dimension 'M' of the key tensor";
      // 检查 attn_bias_shape 的最后一维必须能被 8 整除
      CHECK_EQ_OR_RETURN(attn_bias_shape->At(num_attn_bias_axes - 1) % 8, 0)
          << "The size of the -1 dimension of attn_bias should be a multiple of 8.";
      if (num_attn_bias_axes >= 2) {
        CHECK_OR_RETURN(attn_bias_shape->At(num_attn_bias_axes - 2) == 1
                        || attn_bias_shape->At(num_attn_bias_axes - 2) >= q_m)
            << "The size of the -2 dimension of attn_bias should be greater than or equal to the "
               "dimension 'M' of the query tensor or equal to 1.";
      }
      if (num_attn_bias_axes >= 3) {
        CHECK_OR_RETURN(attn_bias_shape->At(num_attn_bias_axes - 3) == 1
                        || attn_bias_shape->At(num_attn_bias_axes - 3) == q_h)
            << "The size of the -3 dimension of attn_bias should be equal to the dimension 'H' of "
               "the query tensor or equal to 1.";
      }
      if (num_attn_bias_axes == 4) {
        CHECK_OR_RETURN(attn_bias_shape->At(0) == 1 || attn_bias_shape->At(0) == q_b)
            << "The size of the -4 dimension of attn_bias should be equal to the dimension 'B' of "
               "the query tensor or equal to 1.";
      }
    }
    const bool o_bm_packed = output_layout == "(BM)(HK)";
    CHECK_EQ_OR_RETURN(o_bm_packed, q_bm_packed)
        << "The query tensor and the output tensor should either both be BM-Packed or both not be "
           "BM-Packed at the same time.";
    std::string op_output_layout;
    if (output_layout == "BM(HK)" || output_layout == "(BM)(HK)") {
      op_output_layout = output_layout;
    } else if (output_layout == "MB(HK)") {
      if (q_b == 1) {
        op_output_layout = output_layout;
      } else {
        op_output_layout = "BM(HK)";
      }
    } else {
      UNIMPLEMENTED_THEN_RETURN() << "output_layout should be 'BM(HK)', 'MB(HK)' or (BM)(HK)";
    }

    double scale_value = 0.0;
    if (scale) {
      scale_value = JUST(scale);
    } else {
      scale_value = 1.0 / std::sqrt(static_cast<float>(q_k));
    }

    auto& attrs = THREAD_CACHED_MUTABLE_ATTR_MAP("query_layout", "key_layout", "value_layout",
                                                 "output_layout", "query_head_size",
                                                 "attn_mask_type", "causal_diagonal_offset",
                                                 "query_max_seq_len", "key_max_seq_len", "scale");
    attrs.SetAllAttrs(query_layout, key_tensor_layout, value_tensor_layout, op_output_layout, q_k,
                      attn_mask_type_val, causal_diagonal_offset, query_max_seq_len.value_or(0),
                      key_max_seq_len.value_or(0), scale_value);
    OpExprCacheKey cache_key{};
    std::vector<std::shared_ptr<one::Tensor>> inputs;
    inputs.emplace_back(query);
    inputs.emplace_back(key_tensor);
    inputs.emplace_back(value_tensor);
    if (attn_bias) {
      inputs.emplace_back(JUST(attn_bias));
      cache_key.has_attn_bias = true;
    } else {
      cache_key.has_attn_bias = false;
    }
    if (query_seq_start && key_seq_start) {
      inputs.emplace_back(JUST(query_seq_start));
      inputs.emplace_back(JUST(key_seq_start));
      cache_key.has_seq_start = true;
    } else {
      cache_key.has_seq_start = false;
    }
    if (key_seq_len) {
      inputs.emplace_back(JUST(key_seq_len));
      cache_key.has_key_seq_len = true;
    } else {
      cache_key.has_key_seq_len = false;
    }
    auto it = op_cache_.find(cache_key);
    CHECK_OR_RETURN(it != op_cache_.end());
    TensorTuple input_tuple(inputs.size());
    for (int i = 0; i < inputs.size(); ++i) { input_tuple[i] = std::move(inputs[i]); }
    std::shared_ptr<one::Tensor> op_output =
        JUST(OpInterpUtil::Dispatch<Tensor>(*it->second, input_tuple, attrs));
    if (op_output_layout == output_layout) {
      return op_output;
    } else {
      if (op_output_layout == "BM(HK)" && output_layout == "MB(HK)") {
        return functional::Transpose(op_output, {1, 0, 2});
      } else {
        UNIMPLEMENTED_THEN_RETURN();
      }
    }
  }

 private:
  OpExprCache op_cache_;
};

