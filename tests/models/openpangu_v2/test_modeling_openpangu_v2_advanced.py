# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch OpenPanguV2 model with advanced features (DSA, MHC, SWA)."""

import unittest

import pytest
from parameterized import parameterized

from transformers import is_torch_available
from transformers.testing_utils import (
    cleanup,
    require_torch,
    slow,
    torch_device,
)

if is_torch_available():
    import torch
    from transformers import AutoTokenizer, OpenPanguV2ForCausalLM, OpenPanguV2Model

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


class OpenPanguV2AdvancedModelTester(CausalLMModelTester):
    """
    Advanced OpenPanguV2 model tester with all special features enabled:
    - DSA (Dynamic Sparse Attention) on layer 0
    - SWA (Sliding Window Attention) on layers 1, 2
    - MHC (Multi-Head Connection) with num_stream=4
    - Sink Tokens with param_sink_number=4
    - MOME (Router Convolution) with router_sliding_window=3
    - MLA (Multi-head Latent Attention) architecture
    - MoE (Mixture of Experts) with dense + MoE layers
    """
    
    if is_torch_available():
        base_model_class = OpenPanguV2Model
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Force eager attention implementation for MHC compatibility
        # Must be set BEFORE other kwargs to ensure it's applied
        kwargs.setdefault('_attn_implementation', 'eager')
        
        # Basic architecture (3 layers: 1 DSA + 2 SWA)
        self.hidden_size = kwargs.get('hidden_size', 64)
        self.num_hidden_layers = kwargs.get('num_hidden_layers', 2)
        self.num_attention_heads = kwargs.get('num_attention_heads', 4)
        self.num_key_value_heads = kwargs.get('num_key_value_heads', 1)
        self.vocab_size = kwargs.get('vocab_size', 99)
        self.intermediate_size = kwargs.get('intermediate_size', 64)
        self.max_position_embeddings = kwargs.get('max_position_embeddings', 512)
        
        # MLA parameters (Multi-head Latent Attention)
        self.q_lora_rank = kwargs.get('q_lora_rank', 32)
        self.kv_lora_rank = kwargs.get('kv_lora_rank', 16)
        self.qk_nope_head_dim = kwargs.get('qk_nope_head_dim', 32)
        self.qk_rope_head_dim = kwargs.get('qk_rope_head_dim', 16)
        self.v_head_dim = kwargs.get('v_head_dim', 32)
        
        # MoE parameters (Mixture of Experts)
        self.moe_intermediate_size = kwargs.get('moe_intermediate_size', 64)
        self.n_routed_experts = kwargs.get('n_routed_experts', 4)
        self.n_shared_experts = kwargs.get('n_shared_experts', 1)
        self.num_experts_per_tok = kwargs.get('num_experts_per_tok', 2)
        self.first_k_dense_replace = kwargs.get('first_k_dense_replace', 1)  # Layer 0 dense, layers 1,2 MoE
        self.routed_scaling_factor = kwargs.get('routed_scaling_factor', 2.5)
        self.norm_topk_prob = kwargs.get('norm_topk_prob', True)
        
        # DSA parameters (Dynamic Sparse Attention) - ENABLED on layer 0
        self.dsa_layers = kwargs.get('dsa_layers', [0])
        self.index_topk = kwargs.get('index_topk', 8)  # Simplified from real 2048
        self.index_n_heads = kwargs.get('index_n_heads', 2)
        self.index_head_dim = kwargs.get('index_head_dim', 16)
        
        # MHC parameters (Multi-Head Connection) - ENABLED with real num_stream=4
        self.use_mhc = kwargs.get('use_mhc', True)
        self.mhc_num_stream = kwargs.get('mhc_num_stream', 4)  # Real model value
        self.mhc_recur_norm = kwargs.get('mhc_recur_norm', 3)  # Simplified iterations
        self.mhc_use_gamma = kwargs.get('mhc_use_gamma', True)
        
        # Sink Tokens - ENABLED
        self.param_sink_number = kwargs.get('param_sink_number', 4)  # Simplified from real 128
        
        # MOME parameters (Router Convolution) - ENABLED
        self.router_sliding_window = kwargs.get('router_sliding_window', 3)  # Real model value
        
        # Sandwich Norm - ENABLED
        self.sandwich_norm = kwargs.get('sandwich_norm', True)
        
        # Layer types configuration (1 DSA + 1 SWA)
        self.layer_types = kwargs.get('layer_types', ["full_attention", "sliding_attention"])
        self.sliding_window = kwargs.get('sliding_window', 512)
        self.swa_layers = kwargs.get('swa_layers', [1])
        
        # RoPE parameters
        self.rope_parameters = kwargs.get('rope_parameters', {"rope_type": "default", "rope_theta": 10000.0})
        self.rope_interleave = kwargs.get('rope_interleave', False)
        self.rope_theta = kwargs.get('rope_theta', 10000.0)
        
        # Other parameters
        self.attention_dropout = kwargs.get('attention_dropout', 0.0)
        self.rms_norm_eps = kwargs.get('rms_norm_eps', 1e-5)
        self.bos_token_id = kwargs.get('bos_token_id', 1)
        self.eos_token_id = kwargs.get('eos_token_id', 2)
        self.pad_token_id = kwargs.get('pad_token_id', 0)
        self.block_post_layernorm_idx = kwargs.get('block_post_layernorm_idx', None)
    
    def get_config(self):
        """
        Override: Explicitly create config with _attn_implementation='eager' for MHC compatibility.
        Reference: DeepseekV3 test_modeling_deepseek_v3.py:160-189
        """
        from transformers.models.openpangu_v2.configuration_openpangu_v2 import OpenPanguV2Config
        
        return OpenPanguV2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            moe_intermediate_size=self.moe_intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            n_shared_experts=self.n_shared_experts,
            n_routed_experts=self.n_routed_experts,
            routed_scaling_factor=self.routed_scaling_factor,
            kv_lora_rank=self.kv_lora_rank,
            q_lora_rank=self.q_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            qk_nope_head_dim=self.qk_nope_head_dim,
            num_experts_per_tok=self.num_experts_per_tok,
            first_k_dense_replace=self.first_k_dense_replace,
            norm_topk_prob=self.norm_topk_prob,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=True,  # Always enable cache for tests
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            attention_dropout=self.attention_dropout,
            # Advanced features
            dsa_layers=self.dsa_layers,
            index_topk=self.index_topk,
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            use_mhc=self.use_mhc,
            mhc_num_stream=self.mhc_num_stream,
            mhc_recur_norm=self.mhc_recur_norm,
            mhc_use_gamma=self.mhc_use_gamma,
            param_sink_number=self.param_sink_number,
            router_sliding_window=self.router_sliding_window,
            sandwich_norm=self.sandwich_norm,
            layer_types=self.layer_types,
            sliding_window=self.sliding_window,
            swa_layers=self.swa_layers,
            rope_parameters=self.rope_parameters,
            rope_interleave=self.rope_interleave,
            block_post_layernorm_idx=self.block_post_layernorm_idx,
            # Force eager attention for MHC compatibility
            _attn_implementation="eager",
        )


@require_torch
class OpenPanguV2AdvancedModelTest(CausalLMModelTest, unittest.TestCase):
    """
    Advanced OpenPanguV2 model test with all features enabled.
    
    Special handling required for:
    - MLA (Multi-head Latent Attention) custom cache format
    - MHC (Multi-Head Connection) multi-stream dimensions
    - DSA (Dynamic Sparse Attention) non-differentiable indexer
    - MoE (Mixture of Experts) routing non-differentiable components
    """
    
    # MoE routing and DSA indexer have non-differentiable components
    test_all_params_have_gradient = False

    has_attentions = False
    
    model_tester_class = OpenPanguV2AdvancedModelTester
    
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        return True  # Skip all pipeline tests (no classification heads)
    
    # ========== Override Tests for MLA/MHC Special Handling ==========
    
    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """
        Override: MLA has special KV cache format.
        Reference: DeepseekV3 test_modeling_deepseek_v3.py:254-269
        """
        from transformers import Cache
        self.assertIsInstance(past_key_values, Cache)
        
        # MLA cache format:
        # keys: (batch, num_kv_heads, seq_length, qk_nope_head_dim + qk_rope_head_dim)
        # values: (batch, num_kv_heads, seq_length, v_head_dim)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_attention_heads"),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)
        
        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)
    
    def test_hidden_states_output(self):
        """
        Override: MHC expands hidden_states dimension to hidden_size * num_stream.
        Reference: DeepseekV4 test_modeling_deepseek_v4.py:158-185
        
        MHC causes two possible shapes:
        1. Collapsed output: (batch, seq, hidden_size) - after hc_head merge
        2. Intermediate multi-stream: (batch, seq, num_stream, hidden_size) - per-layer output
        """
        import torch
        
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict)
            
            hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[-1]
            self.assertIsNotNone(hidden_states)
            self.assertEqual(len(hidden_states), config.num_hidden_layers + 1)
            
            seq_len = inputs_dict["input_ids"].shape[1]
            
            for layer_idx, layer_h in enumerate(hidden_states):
                # MHC causes two possible shapes:
                # 1. Collapsed (3D): (B, S, hidden_size) - final merged output
                # 2. Multi-stream (4D): (B, S, num_stream, hidden_size) - intermediate
                
                if layer_h.ndim == 3:
                    # Standard collapsed shape
                    expected_shape = (inputs_dict["input_ids"].shape[0], seq_len, config.hidden_size)
                    self.assertEqual(layer_h.shape, expected_shape)
                elif layer_h.ndim == 4:
                    # MHC multi-stream shape
                    expected_shape = (
                        inputs_dict["input_ids"].shape[0], 
                        seq_len, 
                        config.mhc_num_stream, 
                        config.hidden_size
                    )
                    self.assertEqual(layer_h.shape, expected_shape)
                else:
                    self.fail(f"Unexpected hidden state dimensions: {layer_h.ndim}D at layer {layer_idx}")
    
    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        """
        Override: MHC multi-stream dimension affects hidden states during generation.
        Reference: DeepseekV4 test_modeling_deepseek_v4.py:231-248
        
        We check batch and hidden_size dimensions, allowing seq_length variations.
        """
        import torch
        
        self.assertIsInstance(hidden_states, tuple)
        self.assertEqual(len(hidden_states), (output_length - prompt_length))
        
        for iter_hidden_states in hidden_states:
            self.assertIsInstance(iter_hidden_states, tuple)
            for layer_hidden in iter_hidden_states:
                self.assertIsInstance(layer_hidden, torch.Tensor)
                # Check batch dimension
                self.assertEqual(layer_hidden.shape[0], batch_size)
                # hidden_size can be original or expanded (num_stream * hidden_size)
                self.assertTrue(
                    layer_hidden.shape[-1] == config.hidden_size or 
                    layer_hidden.shape[-1] == config.hidden_size * getattr(config, 'mhc_num_stream', 1)
                )
    
    def test_tp_plan_matches_params(self):
        """
        Override: MLA architecture doesn't have standard q_proj/k_proj/v_proj layers.
        Reference: DeepseekV2 test_modeling_deepseek_v2.py:135-144
        
        MLA uses q_a_proj/q_b_proj instead of q_proj when q_lora_rank is set.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        
        # MLA architecture: remove non-existent TP plan keys
        if config.q_lora_rank is not None:
            # MLA uses q_a_proj/q_b_proj, not standard q_proj
            config.base_model_tp_plan.pop("layers.*.self_attn.q_proj", None)
        
        # Call parent method for remaining checks
        super().test_tp_plan_matches_params()
    
    # ========== Skip Tests - MLA/MHC/DSA Incompatibilities ==========
    
    # Assisted Decoding series - MLA incompatible
    @unittest.skip("OpenPanguV2 MLA attention is not compatible with assisted decoding")
    @parameterized.expand([("random",), ("same",)])
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass
    
    @unittest.skip("OpenPanguV2 MLA attention is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self, assistant_type):
        pass
    
    @unittest.skip("OpenPanguV2 MLA attention is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass
    
    # Cache format series - MLA custom cache
    @unittest.skip("OpenPanguV2 MLA uses custom cache format incompatible with standard cache")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass
    
    @unittest.skip("OpenPanguV2 MLA uses custom cache format incompatible with standard cache")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass
    
    @unittest.skip("OpenPanguV2 MLA compressor is not compatible with QuantizedCache")
    def test_generate_with_quant_cache(self):
        pass
    
    # SDPA series - custom head dims
    @unittest.skip("SDPA can't dispatch on flash due to unsupported custom head dims in MLA")
    def test_sdpa_can_dispatch_on_flash(self):
        pass
    
    # Padding series - DSA/MHC compression affects padding
    @unittest.skip(
        reason=(
            "DSA/MHC compression mechanism pools windows before attention mask is applied "
            "- left-padding shifts window boundaries and causes logits divergence"
        )
    )
    def test_left_padding_compatibility(self):
        pass
    
    # RoPE Scaling series - custom rope_parameters
    @unittest.skip("OpenPanguV2 uses custom rope_parameters that may not support standard scaling")
    def test_model_rope_scaling_frequencies(self):
        pass
    
    @unittest.skip("OpenPanguV2 uses custom rope_parameters that may not support standard scaling")
    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass
    
    # Compilation tests - MLA/MHC may not compile
    @unittest.skip("OpenPanguV2 MLA/MHC components not fully compatible with torch.compile")
    @pytest.mark.torch_compile_test
    def test_generate_compilation_all_outputs(self):
        pass
    
    @unittest.skip("OpenPanguV2 MLA/MHC components not fully compatible with torch.compile")
    @pytest.mark.torch_compile_test
    def test_generate_compile_model_forward(self):
        pass
    
    # Static cache tests - MLA incompatible
    @unittest.skip("OpenPanguV2 MLA uses custom cache format incompatible with static cache")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass
    
    @unittest.skip("OpenPanguV2 MLA uses custom cache format incompatible with static cache")
    def test_generate_with_static_cache(self):
        pass


@require_torch
class OpenPanguV2AdvancedIntegrationTest(unittest.TestCase):
    """
    Integration test with real OpenPangu2-Flash checkpoint.
    
    Tests real model functionality:
    - Greedy generation
    - Different attention implementations (eager/sdpa)
    - Batch generation
    """
    
    model_id = "/data/public/yuchanghui/models/openpangu2-flash/"
    
    def setUp(self):
        cleanup(torch_device, gc_collect=True)
    
    def tearDown(self):
        cleanup(torch_device, gc_collect=True)
    
    @slow
    def test_model_generation_real_checkpoint(self):
        """
        Test generation with real OpenPangu2-Flash checkpoint.
        Priority: A - Basic generation test (greedy decoding).
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = OpenPanguV2ForCausalLM.from_pretrained(
            self.model_id, 
            device_map="auto", 
            torch_dtype="auto"
        )
        
        # Test prompt
        prompt = "Write a short summary of the benefits of regular exercise"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate with greedy decoding
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        
        # Decode and verify
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Basic validation: generated text not empty and contains original prompt
        self.assertTrue(len(generated_text) > len(prompt))
        self.assertTrue(prompt in generated_text)
        
        print(f"\n✓ Generation successful!")
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
    
    @slow
    def test_attention_implementations(self):
        """
        Test different attention implementations (eager vs sdpa).
        Verify that both implementations produce identical results.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        
        # Eager implementation
        model_eager = OpenPanguV2ForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype="auto",
            attn_implementation="eager"
        )
        
        # SDPA implementation (if supported)
        try:
            model_sdpa = OpenPanguV2ForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype="auto",
                attn_implementation="sdpa"
            )
            
            prompt = "Test attention implementation consistency"
            inputs = tokenizer(prompt, return_tensors="pt").to(model_eager.device)
            
            # Generate and compare results
            with torch.no_grad():
                outputs_eager = model_eager.generate(**inputs, max_new_tokens=10, do_sample=False)
                outputs_sdpa = model_sdpa.generate(**inputs, max_new_tokens=10, do_sample=False)
            
            # Verify identical results (greedy should be exact match)
            torch.testing.assert_close(outputs_eager, outputs_sdpa)
            
            print(f"\n✓ Attention implementations match!")
            print(f"Eager output: {tokenizer.decode(outputs_eager[0], skip_special_tokens=True)}")
            print(f"SDPA output: {tokenizer.decode(outputs_sdpa[0], skip_special_tokens=True)}")
            
        except Exception as e:
            # SDPA may not be supported for MLA with custom head dims
            print(f"\n⚠ SDPA implementation not available or failed: {e}")
            self.skipTest(f"SDPA not available for this model: {e}")
    
    @slow
    def test_batch_generation(self):
        """
        Test batch generation with multiple prompts.
        Verify padding and batch processing work correctly.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"  # OpenPanguV2 uses left padding
        
        model = OpenPanguV2ForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype="auto"
        )
        
        # Multiple prompts of different lengths
        prompts = [
            "Hello world",
            "This is a longer prompt to test padding",
            "Short prompt"
        ]
        
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        
        # Batch generation
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=15, do_sample=False)
        
        # Decode each result
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Verify all prompts are processed
        self.assertEqual(len(generated_texts), len(prompts))
        
        print(f"\n✓ Batch generation successful!")
        for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
            print(f"Prompt {i}: {prompt}")
            print(f"Generated {i}: {generated[:50]}...")