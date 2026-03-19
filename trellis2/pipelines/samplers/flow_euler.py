from typing import *
import os
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        emit_timing = getattr(self, "_ai3d_emit_timing", None)
        mark_sample_once = bool(getattr(self, "_ai3d_mark_sample_once", False))
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        if mark_sample_once and callable(emit_timing):
            try:
                emit_timing("pipeline_tex_slat_sample_once_model_call_start", 0)
            except Exception:
                pass
        model_output = model(x_t, t, cond, **kwargs)
        if mark_sample_once and callable(emit_timing):
            try:
                emit_timing("pipeline_tex_slat_sample_once_model_call_returned", 0)
            except Exception:
                pass
        return model_output

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        emit_timing = getattr(self, "_ai3d_emit_timing", None)
        mark_sample_once = bool(getattr(self, "_ai3d_mark_sample_once", False))
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        if mark_sample_once and callable(emit_timing):
            try:
                emit_timing("pipeline_tex_slat_sample_once_model_output_accessed", 0)
            except Exception:
                pass
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        if mark_sample_once and callable(emit_timing):
            try:
                emit_timing("pipeline_tex_slat_sample_once_v_to_xstart_ready", 0)
            except Exception:
                pass
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        emit_timing = getattr(self, "_ai3d_emit_timing", None)
        mark_sample_once = bool(getattr(self, "_ai3d_mark_sample_once", False))
        if mark_sample_once:
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_first_marker_r3\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_entry_s0\n")
            except Exception:
                pass
        if mark_sample_once and callable(emit_timing):
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_get_model_prediction_marker_s1\n")
            except Exception:
                pass
            try:
                emit_timing("pipeline_tex_slat_sample_once_get_model_prediction_start", 0)
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_get_model_prediction_marker_s2\n")
            except Exception:
                pass
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        if mark_sample_once:
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_get_model_prediction_call_s3\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_get_model_prediction_return_t0\n")
            except Exception:
                pass
        if mark_sample_once and callable(emit_timing):
            try:
                emit_timing(
                    "pipeline_tex_slat_sample_once_model_prediction_ready",
                    0,
                    {
                        "predX0Shape": tuple(int(v) for v in getattr(pred_x_0, "shape", ())),
                        "predVShape": tuple(int(v) for v in getattr(pred_v, "shape", ())),
                    },
                )
            except Exception:
                pass

        if mark_sample_once:
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_first_pred_v_access_t1\n")
            except Exception:
                pass
        use_in_place_pred_v = all(hasattr(pred_v, attr) for attr in ("mul_", "neg_", "add_"))
        if mark_sample_once:
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_first_pred_v_access_t2\n")
            except Exception:
                pass

        pred_eps = None
        if mark_sample_once:
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_pred_x_prev_math_t3\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_pred_x_prev_math_u0\n")
            except Exception:
                pass
        if use_in_place_pred_v:
            pred_x_prev = pred_v.mul_(t - t_prev).neg_().add_(x_t)
        else:
            pred_x_prev = x_t - (t - t_prev) * pred_v
        if mark_sample_once:
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_pred_x_prev_math_u1\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_u1_w0\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_emit_timing_condition_w1\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_w1_x0\n")
            except Exception:
                pass
            try:
                emit_timing_local = emit_timing
            except Exception:
                emit_timing_local = None
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_emit_timing_bind_x1\n")
            except Exception:
                pass
            try:
                emit_timing_is_callable = callable(emit_timing_local)
            except Exception:
                emit_timing_is_callable = False
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_emit_timing_callable_eval_x2\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_emit_timing_branch_x3\n")
            except Exception:
                pass
        else:
            emit_timing_local = emit_timing
            emit_timing_is_callable = False
        if mark_sample_once:
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_mark_sample_once_branch_y0\n")
            except Exception:
                pass
        if mark_sample_once:
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_mark_sample_once_branch_entry_y1\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_emit_timing_is_callable_branch_y2\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_emit_timing_flag_assign_z0\n")
            except Exception:
                pass
            emit_timing_flag = emit_timing_is_callable
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_emit_timing_flag_assign_z1\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_emit_timing_flag_branch_z2\n")
            except Exception:
                pass
            if emit_timing_flag:
                try:
                    os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_emit_timing_flag_branch_entry_z3\n")
                except Exception:
                    pass
                try:
                    os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_emit_timing_is_callable_branch_entry_y3\n")
                except Exception:
                    pass
                try:
                    os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_first_statement_in_emit_timing_branch_y4\n")
                except Exception:
                    pass
                try:
                    os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_emit_timing_branch_entry_x4\n")
                except Exception:
                    pass
                try:
                    os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_first_statement_in_emit_timing_branch_y5\n")
                except Exception:
                    pass
                try:
                    os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_emit_timing_block_entry_w2\n")
                except Exception:
                    pass
                try:
                    os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_first_statement_in_emit_timing_block_w3\n")
                except Exception:
                    pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_first_pred_x_prev_access_u2\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_first_pred_x_prev_access_v0\n")
            except Exception:
                pass
            try:
                pred_x_prev_shape = tuple(int(v) for v in getattr(pred_x_prev, "shape", ()))
            except Exception:
                pred_x_prev_shape = tuple()
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_first_pred_x_prev_access_v1\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_before_second_pred_x_prev_access_v2\n")
            except Exception:
                pass
            try:
                pred_x_prev_device = str(getattr(pred_x_prev, "device", "unknown"))
            except Exception:
                pred_x_prev_device = "unknown"
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_second_pred_x_prev_access_v3\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sample_once_after_first_pred_x_prev_access_u3\n")
            except Exception:
                pass
            try:
                emit_timing(
                    "pipeline_tex_slat_sample_once_pred_x_prev_ready",
                    0,
                    {
                        "predXPrevShape": pred_x_prev_shape,
                        "predXPrevDevice": pred_x_prev_device,
                    },
                )
            except Exception:
                pass

        ret = edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})
        if mark_sample_once and callable(emit_timing):
            try:
                emit_timing("pipeline_tex_slat_sample_once_return_ready", 0)
            except Exception:
                pass
        return ret

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        emit_timing = getattr(self, "_ai3d_emit_timing", None)
        keep_prediction_history = bool(getattr(self, "_ai3d_keep_prediction_history", True))
        disable_progress_wrapper = bool(getattr(self, "_ai3d_disable_tqdm", False))
        effective_verbose = bool(verbose) and not disable_progress_wrapper
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})

        if callable(emit_timing):
            try:
                emit_timing(
                    "pipeline_tex_slat_sampler_pre_loop",
                    0,
                    {
                        "samplerClass": type(self).__name__,
                        "disableProgressWrapper": disable_progress_wrapper,
                        "requestedVerbose": bool(verbose),
                        "effectiveVerbose": effective_verbose,
                        "steps": steps,
                    },
                )
            except Exception:
                pass
        try:
            print(
                "[ai3d] pipeline_tex_slat_sampler_pre_loop",
                {
                    "samplerClass": type(self).__name__,
                    "disableProgressWrapper": disable_progress_wrapper,
                    "requestedVerbose": bool(verbose),
                    "effectiveVerbose": effective_verbose,
                    "steps": steps,
                },
            )
        except Exception:
            pass

        def finalize_step(out, is_last_step: bool):
            nonlocal sample
            if is_last_step and callable(emit_timing):
                try:
                    emit_timing(
                        "pipeline_tex_slat_sampler_last_iter_out_ready",
                        0,
                        {
                            "predXPrevShape": tuple(int(v) for v in getattr(getattr(out, "pred_x_prev", None), "shape", ())),
                            "predX0Shape": tuple(int(v) for v in getattr(getattr(out, "pred_x_0", None), "shape", ())),
                        },
                    )
                except Exception:
                    pass
            sample = out.pred_x_prev
            if is_last_step and callable(emit_timing):
                try:
                    emit_timing(
                        "pipeline_tex_slat_sampler_last_iter_sample_assigned",
                        0,
                        {
                            "sampleDevice": str(getattr(sample, "device", "unknown")),
                            "sampleShape": tuple(int(v) for v in getattr(sample, "shape", ())),
                        },
                    )
                except Exception:
                    pass
            if keep_prediction_history:
                ret.pred_x_t.append(out.pred_x_prev)
                ret.pred_x_0.append(out.pred_x_0)
            elif is_last_step:
                pred_x_0 = getattr(out, "pred_x_0", None)
                if torch.is_tensor(pred_x_0):
                    out.pred_x_0 = None
                if callable(emit_timing):
                    try:
                        emit_timing("pipeline_tex_slat_sampler_last_iter_aux_released", 0)
                    except Exception:
                        pass
            if is_last_step and callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_last_iter_pre_del", 0)
                except Exception:
                    pass
            del out
            if is_last_step and callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_last_iter_post_del", 0)
                except Exception:
                    pass
            if is_last_step and not keep_prediction_history and torch.cuda.is_available():
                try:
                    import gc

                    gc.collect()
                except Exception:
                    pass
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                if callable(emit_timing):
                    try:
                        emit_timing("pipeline_tex_slat_sampler_last_iter_flush_complete", 0)
                    except Exception:
                        pass

        regular_pairs = t_pairs[:-1]
        final_pair = t_pairs[-1] if t_pairs else None

        regular_iter = regular_pairs
        if effective_verbose:
            regular_iter = tqdm(regular_pairs, desc=tqdm_desc, disable=False)

        for t, t_prev in regular_iter:
            self._ai3d_mark_sample_once = False
            try:
                out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            finally:
                self._ai3d_mark_sample_once = False
            finalize_step(out, False)

        if callable(emit_timing):
            try:
                emit_timing(
                    "pipeline_tex_slat_sampler_post_loop",
                    0,
                    {
                        "samplerClass": type(self).__name__,
                        "disableProgressWrapper": disable_progress_wrapper,
                        "effectiveVerbose": effective_verbose,
                        "regularStepCount": len(regular_pairs),
                    },
                )
            except Exception:
                pass
        try:
            print(
                "[ai3d] pipeline_tex_slat_sampler_post_loop",
                {
                    "samplerClass": type(self).__name__,
                    "disableProgressWrapper": disable_progress_wrapper,
                    "effectiveVerbose": effective_verbose,
                    "regularStepCount": len(regular_pairs),
                },
            )
        except Exception:
            pass

        if callable(emit_timing):
            try:
                emit_timing(
                    "pipeline_tex_slat_sampler_gap_after_post_loop",
                    0,
                    {
                        "samplerClass": type(self).__name__,
                        "effectiveVerbose": effective_verbose,
                        "keepPredictionHistory": keep_prediction_history,
                    },
                )
            except Exception:
                pass
        try:
            print(
                "[ai3d] pipeline_tex_slat_sampler_gap_after_post_loop",
                {
                    "samplerClass": type(self).__name__,
                    "effectiveVerbose": effective_verbose,
                    "keepPredictionHistory": keep_prediction_history,
                },
            )
        except Exception:
            pass

        if effective_verbose and hasattr(regular_iter, "close"):
            try:
                regular_iter.close()
            except Exception:
                pass
        try:
            del regular_iter
        except Exception:
            pass

        if not keep_prediction_history and torch.cuda.is_available():
            try:
                import gc

                gc.collect()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

        if callable(emit_timing):
            try:
                emit_timing(
                    "pipeline_tex_slat_sampler_gap_after_cleanup",
                    0,
                    {
                        "samplerClass": type(self).__name__,
                        "effectiveVerbose": effective_verbose,
                        "keepPredictionHistory": keep_prediction_history,
                    },
                )
            except Exception:
                pass
        try:
            print(
                "[ai3d] pipeline_tex_slat_sampler_gap_after_cleanup",
                {
                    "samplerClass": type(self).__name__,
                    "effectiveVerbose": effective_verbose,
                    "keepPredictionHistory": keep_prediction_history,
                },
            )
        except Exception:
            pass

        if callable(emit_timing):
            try:
                emit_timing(
                    "pipeline_tex_slat_sampler_gap_before_pre_separate",
                    0,
                    {
                        "hasFinalPair": final_pair is not None,
                        "samplerClass": type(self).__name__,
                    },
                )
            except Exception:
                pass
        try:
            print(
                "[ai3d] pipeline_tex_slat_sampler_gap_before_pre_separate",
                {
                    "hasFinalPair": final_pair is not None,
                    "samplerClass": type(self).__name__,
                },
            )
        except Exception:
            pass

        if final_pair is not None:
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_branch_entered", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_branch_entered")
            except Exception:
                pass
            final_t, final_t_prev = final_pair
            if callable(emit_timing):
                try:
                    emit_timing(
                        "pipeline_tex_slat_sampler_gap_final_pair_unpacked",
                        0,
                        {
                            "samplerClass": type(self).__name__,
                            "keepPredictionHistory": keep_prediction_history,
                            "historyXT": len(getattr(ret, "pred_x_t", [])),
                            "historyX0": len(getattr(ret, "pred_x_0", [])),
                        },
                    )
                except Exception:
                    pass
            try:
                print(
                    "[ai3d] pipeline_tex_slat_sampler_gap_final_pair_unpacked",
                    {
                        "samplerClass": type(self).__name__,
                        "keepPredictionHistory": keep_prediction_history,
                        "historyXT": len(getattr(ret, "pred_x_t", [])),
                        "historyX0": len(getattr(ret, "pred_x_0", [])),
                    },
                )
            except Exception:
                pass

            del final_pair
            try:
                del regular_pairs
            except Exception:
                pass
            try:
                del t_pairs
            except Exception:
                pass
            try:
                del t_seq
            except Exception:
                pass
            if callable(emit_timing):
                try:
                    emit_timing(
                        "pipeline_tex_slat_sampler_gap_after_sequence_release",
                        0,
                        {
                            "samplerClass": type(self).__name__,
                            "hasSample": sample is not None,
                        },
                    )
                except Exception:
                    pass
            try:
                print(
                    "[ai3d] pipeline_tex_slat_sampler_gap_after_sequence_release",
                    {
                        "samplerClass": type(self).__name__,
                        "hasSample": sample is not None,
                    },
                )
            except Exception:
                pass

            if keep_prediction_history:
                try:
                    ret.pred_x_t = []
                    ret.pred_x_0 = []
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        import gc

                        gc.collect()
                    except Exception:
                        pass
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            if callable(emit_timing):
                try:
                    emit_timing(
                        "pipeline_tex_slat_sampler_gap_after_history_release",
                        0,
                        {
                            "samplerClass": type(self).__name__,
                            "historyXT": len(getattr(ret, "pred_x_t", [])),
                            "historyX0": len(getattr(ret, "pred_x_0", [])),
                        },
                    )
                except Exception:
                    pass
                try:
                    print(
                        "[ai3d] pipeline_tex_slat_sampler_gap_after_history_release",
                        {
                            "samplerClass": type(self).__name__,
                            "historyXT": len(getattr(ret, "pred_x_t", [])),
                            "historyX0": len(getattr(ret, "pred_x_0", [])),
                        },
                    )
                except Exception:
                    pass

            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_after_history_release_marker_e", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_after_history_release_marker_e")
            except Exception:
                pass

            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_before_pre_pre_separate_marker_f", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_before_pre_pre_separate_marker_f")
            except Exception:
                pass

            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_pre_pre_separate_marker", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_pre_pre_separate_marker")
            except Exception:
                pass

            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_pre_separate_last_iter", 0)
                except Exception:
                    pass
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_post_pre_separate_marker", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_post_pre_separate_marker")
            except Exception:
                pass
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_after_post_pre_separate_marker_i", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_after_post_pre_separate_marker_i")
            except Exception:
                pass
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_before_marker_g_j", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_before_marker_g_j")
            except Exception:
                pass
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_before_final_t_assignment_marker_g", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_before_final_t_assignment_marker_g")
            except Exception:
                pass
            t, t_prev = final_t, final_t_prev
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_after_final_t_assignment_marker_h", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_after_final_t_assignment_marker_h")
            except Exception:
                pass
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_after_marker_h_m", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_after_marker_h_m")
            except Exception:
                pass
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_before_marker_k_n", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_before_marker_k_n")
            except Exception:
                pass
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_gap_after_final_t_assignment_marker_k", 0)
                except Exception:
                    pass
            try:
                print("[ai3d] pipeline_tex_slat_sampler_gap_after_final_t_assignment_marker_k")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_gap_before_marker_l0\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_gap_before_separate_last_iter_marker_l1\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_gap_after_marker_l_o\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_separate_last_iter_entered_p0\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_separate_last_iter_entered\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_separate_last_iter_entered_p1\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_separate_last_iter_after_p1_q0\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_separate_last_iter_before_mark_sample_once_q1\n")
            except Exception:
                pass
            self._ai3d_mark_sample_once = True
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_separate_last_iter_after_mark_sample_once_q2\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_separate_last_iter_before_pre_sample_once_q3\n")
            except Exception:
                pass
            try:
                os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_separate_last_iter_after_q3_r0\n")
            except Exception:
                pass
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_separate_last_iter_mark_set", 0)
                except Exception:
                    pass
            try:
                if callable(emit_timing):
                    try:
                        emit_timing("pipeline_tex_slat_sampler_separate_last_iter_pre_sample_once", 0)
                    except Exception:
                        pass
                try:
                    os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_separate_last_iter_before_sample_once_call_r1\n")
                except Exception:
                    pass
                out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
                try:
                    os.write(2, b"[ai3d-raw] pipeline_tex_slat_sampler_separate_last_iter_after_sample_once_call_r2\n")
                except Exception:
                    pass
                if callable(emit_timing):
                    try:
                        emit_timing("pipeline_tex_slat_sampler_separate_last_iter_sample_once_returned", 0)
                    except Exception:
                        pass
            finally:
                self._ai3d_mark_sample_once = False
            finalize_step(out, True)
        if callable(emit_timing):
            try:
                emit_timing(
                    "pipeline_tex_slat_sampler_loop_complete",
                    0,
                    {
                        "keepPredictionHistory": keep_prediction_history,
                        "sampleDevice": str(getattr(sample, "device", "unknown")),
                        "sampleShape": tuple(int(v) for v in getattr(sample, "shape", ())),
                    },
                )
            except Exception:
                pass
        if not keep_prediction_history:
            ret.pred_x_t = []
            ret.pred_x_0 = []
            if callable(emit_timing):
                try:
                    emit_timing("pipeline_tex_slat_sampler_history_skipped", 0)
                except Exception:
                    pass
        ret.samples = sample
        if callable(emit_timing):
            try:
                emit_timing(
                    "pipeline_tex_slat_sampler_samples_assigned",
                    0,
                    {
                        "sampleDevice": str(getattr(ret.samples, "device", "unknown")),
                        "sampleShape": tuple(int(v) for v in getattr(ret.samples, "shape", ())),
                    },
                )
            except Exception:
                pass
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            guidance_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)
