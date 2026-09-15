"""Frozen, shared-trajectory replay with explicit episode/validity provenance."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .locomotion_diagnostics import VERSION, input_validity, transition_diagnostic, TRANSITION_RADIUS


def replay_episode(logits, probabilities, valid, classes, ema, bayes, canonicalize):
    """No update for invalid input; hold each method's last selected skill."""
    methods=("instantaneous","ema","bayes")
    selected={m:[] for m in methods}; held={m:"rough" for m in methods}
    ema_values=[]; beliefs=[]
    last_ema=torch.full_like(logits[0],float("nan"))
    last_belief=torch.full_like(probabilities[0],1/len(classes))
    for i in range(len(logits)):
        if valid[i]:
            held["instantaneous"]=canonicalize(classes[int(logits[i].argmax())])
            held["ema"]=canonicalize(ema.update(logits[i]))
            last_ema=ema.ema_scores.detach().clone()
            step=bayes.update(probabilities[i]); held["bayes"]=canonicalize(step.label)
            last_belief=step.posterior.detach().clone()
        for m in methods: selected[m].append(held[m])
        ema_values.append(last_ema.cpu()); beliefs.append(last_belief.cpu())
    return selected, torch.stack(ema_values), torch.stack(beliefs)


def run_replay(args, payloads, runtimes, make_bayes, ema_class, canonicalize, balanced_accuracy, baselines):
    output=Path(args.output)/"replay_diagnostic_bundles"
    output.mkdir(parents=True,exist_ok=True)
    predictions_out=[]; transitions_out=[]; groups=defaultdict(list); unavailable=[]; bundle_paths=[]
    for payload in sorted(payloads,key=lambda p:p["_path"]):
        meta=payload["metadata"]; source=meta["paper_method"]
        if source=="distilled": continue
        if not (payload.get("trajectory_replay") or {}).get("files"):
            unavailable.append(dict(run=payload["_path"],reason="No saved replay observations"));continue
        layout={int(t["track_id"]):t["sequence"] for t in meta["track_layout"]}
        for entry in payload["trajectory_replay"]["files"]:
            path=Path(entry["path"])
            if not path.exists():
                relocated=Path(payload["_path"]).parent/path.parent.name/path.name
                if relocated.exists(): path=relocated
            data=torch.load(path,map_location="cpu",weights_only=False)
            n=len(data["depth"])
            if "episode_id" not in data:
                unavailable.append(dict(run=payload["_path"],reason="Legacy replay lacks verified episode IDs; not joined/replayed"));continue
            episode_ids=data["episode_id"]
            if not n: continue
            # Each contiguous episode must occur only once in the file.
            starts=[0]+[i for i in range(1,n) if episode_ids[i]!=episode_ids[i-1]]+[n]
            assert len({episode_ids[i] for i in starts[:-1]})==len(starts)-1
            for lo,hi in zip(starts[:-1],starts[1:]):
                source_data={k:(v[lo:hi] if (torch.is_tensor(v) or isinstance(v,list)) and len(v)==n else v)
                             for k,v in data.items()}
                valid=input_validity(source_data["depth"],source_data["orientation_rpy"],source_data["angular_velocity"]).numpy()
                assert np.array_equal(valid,source_data["input_valid"])
                times=np.array(source_data["timestamp_s"],float); positions=source_data["base_position"][:,0].numpy()
                assert np.all(np.diff(times)>0)
                assert source_data["reset_boundary"][0] and not any(source_data["reset_boundary"][1:])
                track=int(entry["track_id"]); seq=layout[track]
                truth=[canonicalize(t) for t in source_data["ground_truth"]]
                edges=np.r_[0,np.flatnonzero(np.array(truth[1:])!=np.array(truth[:-1]))+1,len(truth)]
                window=np.zeros(len(truth),bool)
                for t in edges[1:-1]: window[max(0,t-TRANSITION_RADIUS):min(len(truth),t+TRANSITION_RADIUS+1)]=True
                events=source_data["transition_events"][0]
                common=dict(metric_version=VERSION,source_method=source,source_run=payload["_path"],
                    difficulty_level=meta["difficulty_level"],evaluation_seed=meta["seed"],
                    track_id=track,episode_id=episode_ids[lo],episode_index=source_data["episode_index"][0],
                    terminal_outcome=source_data["terminal_outcome"][0])
                saved=dict(metadata=common,source=source_data,truth=truth,transition_mask=window,
                           validity=valid,architectures={},transitions=[])
                for arch,(runtime,manifest) in runtimes.items():
                    classes=list(runtime.class_ids)
                    device=getattr(getattr(runtime,"classifier",runtime),"device","cpu")
                    # Frozen batched inference only on the same valid observations as online.
                    ids=np.flatnonzero(valid); logits=torch.full((len(truth),len(classes)),float("nan"))
                    probs=logits.clone()
                    for start in range(0,len(ids),256):
                        ix=torch.as_tensor(ids[start:start+256],dtype=torch.long)
                        z,q=runtime.predict_deterministic(source_data["depth"][ix].to(device),
                            source_data["orientation_rpy"][ix].to(device),source_data["angular_velocity"][ix].to(device),
                            temperature=manifest["fixed_bayes_configuration"]["T_filter"])
                        logits[ix]=z.detach().cpu();probs[ix]=q.detach().cpu()
                    ema=ema_class(classes,**manifest["fixed_ema_configuration"],device=device)
                    bayes=make_bayes(classes,manifest["fixed_bayes_configuration"],1,device)[0]
                    selected,ema_scores,beliefs=replay_episode(logits,probs,valid,classes,ema,bayes,canonicalize)
                    source_arch = "raw_depth_nn" if source.startswith("raw_depth_") else "feature_nn"
                    source_temporal = source.rsplit("_",1)[-1]
                    online_check = None
                    if source != "oracle" and arch == source_arch and source_temporal in selected:
                        observed = source_data["selected_skill"]
                        online_check = dict(compared_ticks=len(observed), mismatched_ticks=sum(
                            a != b for a,b in zip(observed,selected[source_temporal])))
                    saved["architectures"][arch]=dict(class_ordering=classes,logits=logits,
                        probabilities=probs,ema_scores=ema_scores,ema_probabilities=ema_scores.softmax(-1),
                        bayes_beliefs=beliefs,selected=selected,model_seed=args.selected_classifier_seeds[arch],
                        online_replay_consistency=online_check,
                        filter_settings={k:manifest[k] for k in ("fixed_ema_configuration","fixed_bayes_configuration")})
                    for method in ("instantaneous","ema","bayes"):
                        pred=selected[method]
                        group=(arch,method,source,meta["difficulty_level"])
                        groups[group].append((truth,pred,window,valid))
                        groups[(arch,method,source,"all")].append((truth,pred,window,valid))
                        for i in range(len(truth)):
                            nearest=min(events,key=lambda e:abs(positions[i]-e["boundary_position_m"])) if events else None
                            event_time=nearest["boundary_timestamp_s"] if nearest else None
                            canon_probs={c:0. for c in ("rough","gap","pit","stairs")}
                            for j,c in enumerate(classes): canon_probs[canonicalize(c)]+=float(probs[i,j])
                            row=dict(common,architecture=arch,temporal_method=method,classifier_seed=args.selected_classifier_seeds[arch],
                                frame_index=i,control_step=source_data["control_step"][i],timestamp_s=times[i],
                                forward_position_m=float(positions[i]),ground_truth=truth[i],predicted_skill=pred[i],
                                input_valid=bool(valid[i]),class_probabilities=canon_probs,
                                logits=logits[i].tolist(),ema_scores=ema_scores[i].tolist(),bayes_beliefs=beliefs[i].tolist(),
                                in_transition_window=bool(window[i]),
                                transition_pair=nearest["transition_pair"] if nearest else None,
                                raw_transition_pair=nearest["raw_transition_pair"] if nearest else None,
                                boundary_position_m=nearest["boundary_position_m"] if nearest else None,
                                boundary_timestamp_s=event_time,
                                signed_distance_to_boundary_m=float(positions[i]-nearest["boundary_position_m"]) if nearest else None,
                                time_relative_to_boundary_s=float(times[i]-event_time) if event_time is not None else None,
                                probability_upcoming_skill=canon_probs[nearest["transition_pair"].split("->")[1]] if nearest else None)
                            predictions_out.append(row)
                        # One record per planned boundary, including failed/unreached approaches.
                        for event in events:
                            j=event["boundary_segment"]; target=canonicalize(seq[j+1]);current=canonicalize(seq[j])
                            crossing_time=event["boundary_timestamp_s"]
                            crossing=int(np.searchsorted(times,crossing_time)) if crossing_time is not None else len(times)
                            matching_segments=[(int(a),int(b)) for a,b in zip(edges[:-1],edges[1:])
                                               if truth[a]==target and a <= crossing < b]
                            segment=min(matching_segments,key=lambda ab:abs(ab[0]-crossing)) if matching_segments and crossing_time is not None else None
                            if segment:
                                a,b=segment; prev=max(0,int(edges[max(0,np.searchsorted(edges,a)-1)]))
                                complete=b<len(truth)
                                diag=transition_diagnostic(pred,target,a,b,times,positions,crossing_time,event["boundary_position_m"],
                                    previous_start=prev,complete=complete,valid=valid)
                            else:
                                a,b=len(truth),len(truth)
                                diag=transition_diagnostic(pred,target,a,b,times,positions,crossing_time,event["boundary_position_m"],
                                    reached=crossing_time is not None,complete=False,valid=valid)
                            ix=diag["persistent_switch_index"]
                            local=np.arange(max(0,min(a,len(truth))-TRANSITION_RADIUS),min(len(truth),a+TRANSITION_RADIUS+1))
                            observed=local[valid[local]] if crossing_time is not None and segment else np.array([],dtype=int)
                            row=dict(common,**event,**{k:v for k,v in diag.items() if k!="metric_version"})
                            pre=[i for i in range(max(0,a-TRANSITION_RADIUS),min(a,len(truth))) if valid[i]]
                            post=[i for i in range(a,min(b,a+TRANSITION_RADIUS)) if valid[i]]
                            occupancy=lambda ix,label,equal: float(np.mean([(pred[i]==label)==equal for i in ix])) if ix else None
                            row.update(architecture=arch,temporal_method=method,classifier_seed=args.selected_classifier_seeds[arch],
                                skill_change_required=current!=target,recognition_segment_start=a,recognition_segment_end_exclusive=b,
                                switch_frame_index=ix,switch_timestamp_s=None if ix is None else times[ix],
                                switch_position_m=None if ix is None else float(positions[ix]),
                                transition_window_accuracy=float(np.mean([pred[i]==truth[i] for i in observed])) if len(observed) else None,
                                late_transition=None if ix is None else bool(ix>a+TRANSITION_RADIUS),
                                premature_transition=None if ix is None else bool(ix<a-TRANSITION_RADIUS),
                                pre_transition_wrong_skill_occupancy=occupancy(pre,current,False),
                                pre_transition_upcoming_skill_occupancy=occupancy(pre,target,True),
                                post_transition_wrong_skill_occupancy=occupancy(post,target,False),
                                post_transition_previous_skill_occupancy=occupancy(post,current,True))
                            transitions_out.append(row);saved["transitions"].append(row)
                digest=hashlib.sha256((payload["_path"]+episode_ids[lo]).encode()).hexdigest()[:16]
                out=output/f"trajectory_{digest}.pt";torch.save(saved,out);bundle_paths.append(str(out))
    accuracy=[]
    for (arch,method,source,difficulty), members in sorted(groups.items()):
        y=[];p=[];w=[]
        for truth,pred,mask,valid in members:
            y.extend(np.asarray(truth)[valid]);p.extend(np.asarray(pred)[valid]);w.extend(mask[valid])
        correct=np.asarray(y)==np.asarray(p);w=np.array(w,bool);offline=baselines.get((arch,method),{})
        ba=balanced_accuracy(y,p)
        offline_classes=list(runtimes[arch][1].get("class_ordering",[]))
        mapped=[canonicalize(c) for c in offline_classes]
        same_labels=len(mapped)==len(set(mapped)) and set(mapped)=={"rough","gap","pit","stairs"}
        accuracy.append(dict(architecture=arch,temporal_method=method,source_method=source,difficulty_level=difficulty,
            classifier_seed=args.selected_classifier_seeds[arch],num_frames=len(y),
            accuracy=float(correct.mean()) if len(y) else None,balanced_accuracy=ba,
            transition_window_accuracy=float(correct[w].mean()) if w.any() else None,
            steady_state_accuracy=float(correct[~w].mean()) if (~w).any() else None,
            offline_balanced_accuracy=offline.get("balanced_accuracy"),
            online_minus_offline_balanced_accuracy=ba-offline["balanced_accuracy"] if same_labels and ba is not None and offline.get("balanced_accuracy") is not None else None,
            offline_label_space=offline_classes,offline_comparison_same_label_space=same_labels,
            aggregation="valid classification-frame weighted",metric_version=VERSION))
    pair_groups=defaultdict(list)
    for r in predictions_out:
        if r["input_valid"]:
            key=tuple(r[k] for k in ("source_method","architecture","temporal_method","difficulty_level","transition_pair","terminal_outcome"))
            pair_groups[key+("transition" if r["in_transition_window"] else "steady",)].append(r)
    pair_rows=[]
    for key, rows in pair_groups.items():
        pair_rows.append(dict(zip(("source_method","architecture","temporal_method","difficulty_level","transition_pair","terminal_outcome","observation_region"),key),
            num_frames=len(rows),accuracy=float(np.mean([r["predicted_skill"]==r["ground_truth"] for r in rows])),
            balanced_accuracy=balanced_accuracy([r["ground_truth"] for r in rows],[r["predicted_skill"] for r in rows])))
    (output/"accuracy_by_source_pair.json").write_text(json.dumps(pair_rows,indent=2))
    import csv
    with (Path(args.output)/"replay_accuracy_by_source_pair.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(pair_rows[0]) if pair_rows else ["source_method"])
        writer.writeheader();writer.writerows(pair_rows)
    (output/"manifest.json").write_text(json.dumps(dict(version=VERSION,bundles=bundle_paths,unavailable=unavailable,
        reset_rule="new EMA/Bayes per source episode; invalid input holds state",
        aggregation="valid classification-frame weighted; source routing method kept separate",
        first_match="within recorded GT target segment",persistence="2 consecutive valid ticks; pre-boundary run must survive boundary",
        window="+/-5 classification ticks at within-episode GT changes",offline_comparison="canonical rough/gap/pit/stairs; shared mapping"),indent=2))
    return predictions_out,transitions_out,accuracy
