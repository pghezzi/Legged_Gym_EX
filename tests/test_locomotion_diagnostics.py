"""Focused CPU checks; no simulator, checkpoints, or full experiments."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
DIR=ROOT/"legged_gym/scripts/evaluation"
package=types.ModuleType("_locomotion_test");package.__path__=[str(DIR)]
sys.modules[package.__name__]=package


def load(name):
    spec=importlib.util.spec_from_file_location(f"_locomotion_test.{name}",DIR/f"{name}.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sys.modules[spec.name]=module
    return module


d=load("locomotion_diagnostics"); replay=load("locomotion_replay")
runner=load("run_paper_locomotion_evaluation")


class EMA:
    def __init__(self,classes,**kwargs): self.classes=classes;self.ema_scores=None;self.calls=0
    def update(self,z):
        self.calls+=1
        self.ema_scores=z.clone() if self.ema_scores is None else .6*z+.4*self.ema_scores
        return self.classes[int(self.ema_scores.argmax())]


class Bayes:
    def __init__(self,classes): self.classes=classes;self.calls=0
    def update(self,q):
        self.calls+=1
        return types.SimpleNamespace(label=self.classes[int(q.argmax())],posterior=q.clone())


class Tests(unittest.TestCase):
    def test_quota_reservation_and_reset_isolation(self):
        quota=d.EpisodeQuota([0,0,1,1],1)
        self.assertEqual(quota.active,[True,False,True,False])
        totals=[0]*4
        for tick in range(4):
            mask=list(quota.active)
            for i,ok in enumerate(mask): totals[i]+=ok
            if tick==1: quota.reset(0)
        self.assertEqual(totals,[2,0,4,0])
        self.assertEqual(quota.episode,[1,0,0,0])
        self.assertTrue(quota.active[2])
        quota.reset(1);self.assertFalse(quota.active[1])

    def test_heading_and_lateral_correction_with_wrap(self):
        z=torch.tensor([0.,0.,1.,-1.])
        h=torch.tensor([.3,-.3,0.,0.])
        out=d.heading_feedback(z,h)
        self.assertTrue(torch.equal(out.sign(),torch.tensor([-1.,1.,1.,-1.])))
        out=d.heading_feedback(torch.tensor([.1]),torch.tensor([2*np.pi+.2]))
        self.assertLess(out.item(),0)
        a=d.heading_feedback(torch.tensor([0.]),torch.tensor([3*np.pi/2]))
        self.assertGreater(a.item(),0)

    def transition(self,p,start=3,stop=6,**kwargs):
        return d.transition_diagnostic(p,"gap",start,stop,list(range(len(p))),list(range(len(p))),start,start,**kwargs)

    def test_transient_anticipation_does_not_hide_miss(self):
        r=self.transition(["gap","gap","rough","rough","rough","rough"])
        self.assertTrue(r["missed_transition"])
        self.assertIsNone(r["persistent_switch_index"])
        self.assertEqual(r["transient_early_runs"][0]["reversion_index"],2)

    def test_persistent_anticipation_and_first_match_are_distinct(self):
        r=self.transition(["rough","rough","gap","gap","rough","rough"])
        self.assertEqual(r["first_match_delay_ticks"],0)
        self.assertEqual(r["delta_t_switch_s"],-1)
        self.assertEqual(r["reversion_count"],1)

    def test_segment_end_and_censoring(self):
        r=self.transition(["rough"]*6+["gap"])
        self.assertTrue(r["missed_transition"])
        r=self.transition(["rough"]*6,complete=False)
        self.assertIsNone(r["missed_transition"])
        self.assertEqual(r["recognition_status"],"censored_no_match")
        r=self.transition(["rough"]*6,reached=False)
        self.assertEqual(r["recognition_status"],"unreached")

    def test_transition_attribution_not_episode_success(self):
        events=d.boundary_events([0,9,11,19,21,23],list(range(6)),["rough","gap","pit","stairs"],10,"termination")
        self.assertEqual([e["traversal_outcome"] for e in events],["traversed","failed_after_crossing","unreached"])
        table=runner._transition_pair_rows([dict(method="oracle",difficulty_level="easy",
            terrain_sequence=["rough","gap","pit","stairs"],success=False,transition_events=events)])
        gap=next(r for r in table if r["transition_pair"]=="rough->gap")
        self.assertEqual(gap["success_rate"],0)
        self.assertEqual(gap["traversal_success_rate"],1)
        failed=d.boundary_events([0,8.5,9.5],range(3),["rough","gap"],10,"termination")
        self.assertEqual(failed[0]["traversal_outcome"],"failed_approach")

    def test_invalid_holds_state_and_new_episode_resets(self):
        classes=["rough","gap"]
        z=torch.tensor([[0.,5.],[float("nan"),0.],[5.,0.]])
        e,b=EMA(classes),Bayes(classes)
        selected,scores,beliefs=replay.replay_episode(z,z.softmax(-1),[True,False,True],classes,e,b,lambda x:x)
        self.assertEqual(e.calls,2);self.assertEqual(b.calls,2)
        self.assertEqual(selected["ema"][0],selected["ema"][1])
        torch.testing.assert_close(scores[0],scores[1])
        torch.testing.assert_close(beliefs[0],beliefs[1])
        new,_,_=replay.replay_episode(z[2:],z[2:].softmax(-1),[True],classes,EMA(classes),Bayes(classes),lambda x:x)
        self.assertEqual(new["ema"],["rough"])
        v=d.input_validity(torch.tensor([[[1.]],[[0.]],[[1.]]]),torch.tensor([[0.],[0.],[float("nan")]]))
        self.assertEqual(v.tolist(),[True,False,False])

    def test_shared_replay_bundle_two_episodes(self):
        class Runtime:
            class_ids=["rough","gap","pit","stairs"]
            device="cpu"
            def predict_deterministic(self,depth,rpy,omega,temperature):
                z=torch.zeros((len(depth),4));z[:,0]=depth[:,0,0];return z,z.softmax(-1)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);file=folder/"track.pt"
            data=dict(depth=torch.ones(4,1,1),orientation_rpy=torch.zeros(4,3),angular_velocity=torch.zeros(4,3),
                base_position=torch.tensor([[0.,0,0],[1.,0,0],[0.,0,0],[1.,0,0]]),
                episode_id=["0:0","0:0","0:1","0:1"],episode_index=[0,0,1,1],
                reset_boundary=[True,False,True,False],input_valid=[True]*4,timestamp_s=[1.,2.,3.,4.],
                control_step=[1,2,3,4],ground_truth=["rough"]*4,terminal_outcome=["step_cap"]*4,
                selected_skill=["rough"]*4, transition_events=[d.boundary_events(
                    [0.,1.],[1.,2.],["rough","gap"],2.,"step_cap")]*4)
            torch.save(data,file)
            meta=dict(paper_method="feature_bayes",difficulty_level="easy",seed=101,
                      track_layout=[dict(track_id=0,sequence=["rough","gap"])])
            payload=dict(metadata=meta,_path=str(folder/"result.json"),trajectory_replay=dict(files=[dict(path=str(file),track_id=0)]))
            manifest=dict(fixed_ema_configuration={},fixed_bayes_configuration={"T_filter":1.})
            args=types.SimpleNamespace(output=folder,selected_classifier_seeds=dict(feature_nn=0,raw_depth_nn=0))
            predictions,transitions,accuracy=replay.run_replay(args,[payload],
                {a:(Runtime(),manifest) for a in ("feature_nn","raw_depth_nn")},
                lambda classes,cfg,n,device:[Bayes(classes)],EMA,lambda x:x,runner._balanced_accuracy,{})
            self.assertEqual(len(predictions),24)
            self.assertEqual(len(transitions),12)
            self.assertTrue(all(r["missed_transition"] is None for r in transitions))
            self.assertTrue(all(r["transition_window_accuracy"] is None for r in transitions))
            summaries=runner._summarize_transition_replay(transitions)
            self.assertTrue(all(r["missed_transition_denominator"]==0 for r in summaries))
            bundles=list((folder/"replay_diagnostic_bundles").glob("*.pt"));self.assertEqual(len(bundles),2)
            for path in bundles:
                saved=torch.load(path,weights_only=False)
                self.assertEqual(len(set(saved["source"]["episode_id"])),1)
                self.assertEqual(saved["architectures"]["feature_nn"]["logits"].shape,(2,4))
                self.assertEqual(saved["architectures"]["feature_nn"]["online_replay_consistency"]["mismatched_ticks"],0)


if __name__=="__main__":unittest.main()
