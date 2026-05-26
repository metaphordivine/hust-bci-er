# 2026-05-26 Remote Artifact Archival Handoff

## Scope

This handoff records local archival of remote training artifacts. It does not
promote route status, candidate evidence, or evaluation results.

Checkpoint and run-output binaries remain outside git. The repository only
records the storage layout, archive scope, integrity hashes, and uncovered
machine state.

## Local Layout

Archive root:

```text
E:\hust-bci-er_checkpoints\20260526\remote_checkpoint_archives
E:\hust-bci-er_checkpoints\20260526\remote_essential_archives
```

Each available remote machine has local tar archives and manifests:

```text
<machine>_checkpoint_pull_20260526.tar
<machine>_checkpoint_pull_20260526.manifest.tsv
<machine>_essential_pull_20260526.tar
<machine>_essential_pull_20260526.manifest.tsv
```

The tar files preserve the remote relative tree under `root/autodl-tmp/`.
The manifest TSV columns are:

```text
mtime_epoch	size_bytes	absolute_remote_path
```

## Included Remote Run Families

The checkpoint archive pull intentionally included checkpoint-like files only:

```text
*.pt
*.pth
*.ckpt
*.safetensors
```

Included run roots:

```text
/root/autodl-tmp/full_strict_top7_p1_*_20260526_1144
/root/autodl-tmp/strict_top7_p2_*_20260526_1323
/root/autodl-tmp/fresh10_fullflow_*_20260526_1456
/root/autodl-tmp/fresh10_fullflow_*_20260526_1512
/root/autodl-tmp/moe5_expert_protocol_rung0_20260526_1542
```

The essential archive pull included the selected run directories as tar files,
excluding `__pycache__`, to avoid exploding thousands of small files onto the
local exFAT drive:

```text
/root/autodl-tmp/full_strict_top7_p1_*_20260526_1144
/root/autodl-tmp/strict_top7_p2_*_20260526_1323
/root/autodl-tmp/fresh10_fullflow_*_20260526_1456
/root/autodl-tmp/fresh10_fullflow_*_20260526_1512
/root/autodl-tmp/moe5_expert_protocol_rung0_20260526_1530
/root/autodl-tmp/moe5_expert_protocol_rung0_20260526_1542
```

Excluded by design:

```text
predictions.csv
score_matrix.csv
logs
raw outputs
temporary experiment dumps
```

## Archive Inventory

| Machine | Archive | Size bytes | Manifest rows | SHA256 |
|---|---:|---:|---:|---|
| G | `G_checkpoint_pull_20260526.tar` | 106506240 | 137 | `6E008C8048EE97107EBEFCBEC43FC36B585E06BEA8085C8E314794310F7BC698` |
| H | `H_checkpoint_pull_20260526.tar` | 85104640 | 124 | `E50B9136A413A39490634D74D6364EF76388E60BCB2879CE8F6B2D6BF94C094B` |
| I | `I_checkpoint_pull_20260526.tar` | 88391680 | 114 | `2EC8CA6B43CA110B45A0B4C9511AAFDFC10AC69AF220B07F1E1606D0683FCDF2` |
| J | `J_checkpoint_pull_20260526.tar` | 175616000 | 115 | `DF032C5DE3D4BCCB3ABED35C8E4921CDCFDA5E932E92C23BDD3CA219F8D2F3A1` |
| K | `K_checkpoint_pull_20260526.tar` | 193505280 | 121 | `E3E468FB93E3AC21CCCEA1990F3028AE8F16F59D64EA09C3AC96D754404864CA` |
| L | `L_checkpoint_pull_20260526.tar` | 87961600 | 96 | `23B20F22D58C8AF26FE5248BB8676A6F1958CACAD015B02FD97D8965D7603232` |
| M | `M_checkpoint_pull_20260526.tar` | 78284800 | 93 | `3563459EDFD3D0B9574BC81B2469E7A7FFE4162BB885687A7FB907560E2BABCF` |

Total archived tar size: 821370240 bytes.

## Essential Archive Inventory

| Machine | Archive | Size bytes | Manifest bytes | SHA256 |
|---|---:|---:|---:|---|
| G | `G_essential_pull_20260526.tar` | 348221440 | 15715136 | `3D53BADE8A5BCC03A020E2092C4EE60D2C8E77259ED1F8B89607FEC993E85E4D` |
| H | `H_essential_pull_20260526.tar` | 333352960 | 16662761 | `17480F376204BFC9969A5C650DDD2096E485F5CF35697602E7E5572B995967D5` |
| I | `I_essential_pull_20260526.tar` | 316149760 | 15054691 | `24002F4E0A2809DA8D70F44045A025DADDD50A4D43B3A537C25520E4A735C9AC` |
| J | `J_essential_pull_20260526.tar` | 416286720 | 15979828 | `A2C65441E96FC189606DF47A7D9EE36A9828C0E77AFC8157145AFABBB125BAFB` |
| K | `K_essential_pull_20260526.tar` | 430684160 | 15782949 | `38965DB09953AB043AA5408D9235A95427534933B047D7E4B1629AB4354A1716` |
| L | `L_essential_pull_20260526.tar` | 264028160 | 12034798 | `7C2B006471CF36B6A1E850B7C503681516B96CA6BB9FD829CD11D968CF344F81` |
| M | `M_essential_pull_20260526.tar` | 269056000 | 13383333 | `5E39BDCA09B5661ECF37621DEDA626748651A83B5464BB8C63BEC2D254E8FCE6` |

Total essential tar size: 2377779200 bytes.

## F Machine Status

Machine F was later identified as:

```text
Beijing B / 533
61354a8bc1-b3ec4fbc
ssh -p 20042 root@connect.bjb2.seetacloud.com
```

The machine became reachable after restart. Its F-specific roots were:

```text
/root/autodl-tmp/full_expert_asha_F_slowshare_20260525_2243
/root/autodl-tmp/full_expert_asha_F_slowshare_fix_20260525_2302
/root/autodl-tmp/full_expert_asha_F_tail2312_norebalance_2320
/root/autodl-tmp/full_expert_asha_F_tail2312_rung0
```

Remote tar creation completed for F with 91 checkpoint entries and 44006
essential entries. The local F transfer was still in progress when this handoff
was written, so F is not included in the verified local hash tables above.

## Recovery Notes

To restore one archive into a temporary inspection directory, use a directory
outside the repository and extract there:

```text
mkdir E:\hust-bci-er_checkpoints\20260526\inspect_G
tar -xf E:\hust-bci-er_checkpoints\20260526\remote_checkpoint_archives\G_checkpoint_pull_20260526.tar -C E:\hust-bci-er_checkpoints\20260526\inspect_G
```

Do not commit extracted checkpoints, prediction tables, score matrices, or raw
outputs.
