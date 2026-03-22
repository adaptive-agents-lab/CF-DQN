# Experiment Organization

## Overview

This document organizes all experiments into SLURM job scripts. Each script runs **4 experiments in parallel** (one per GPU) to maintain high SPS (~300 steps/sec).

## Total Experiment Count

- **CF-DQN**: 48 experiments (3 seeds × 2 frequencies × 2 max × 4 games)
- **C51**: 12 experiments (3 seeds × 4 games)
- **DQN**: 12 experiments (3 seeds × 4 games)
- **Total**: 72 experiments across 18 scripts

---

## CF-DQN Experiments (12 scripts)

### Configuration Space
- **Seeds**: 1, 2, 3
- **n_frequencies**: 32, 64
- **freq_max**: 2, 5
- **Games**: Breakout, Asterix, Seaquest, Qbert

### Script Organization

#### CF-DQN Batch 1: `scripts/cf_dqn_batch_01.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Breakout | 1 | 32 | 2 |
| 1 | Breakout | 1 | 32 | 5 |
| 2 | Breakout | 1 | 64 | 2 |
| 3 | Breakout | 1 | 64 | 5 |

#### CF-DQN Batch 2: `scripts/cf_dqn_batch_02.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Breakout | 2 | 32 | 2 |
| 1 | Breakout | 2 | 32 | 5 |
| 2 | Breakout | 2 | 64 | 2 |
| 3 | Breakout | 2 | 64 | 5 |

#### CF-DQN Batch 3: `scripts/cf_dqn_batch_03.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Breakout | 3 | 32 | 2 |
| 1 | Breakout | 3 | 32 | 5 |
| 2 | Breakout | 3 | 64 | 2 |
| 3 | Breakout | 3 | 64 | 5 |

#### CF-DQN Batch 4: `scripts/cf_dqn_batch_04.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Asterix | 1 | 32 | 2 |
| 1 | Asterix | 1 | 32 | 5 |
| 2 | Asterix | 1 | 64 | 2 |
| 3 | Asterix | 1 | 64 | 5 |

#### CF-DQN Batch 5: `scripts/cf_dqn_batch_05.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Asterix | 2 | 32 | 2 |
| 1 | Asterix | 2 | 32 | 5 |
| 2 | Asterix | 2 | 64 | 2 |
| 3 | Asterix | 2 | 64 | 5 |

#### CF-DQN Batch 6: `scripts/cf_dqn_batch_06.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Asterix | 3 | 32 | 2 |
| 1 | Asterix | 3 | 32 | 5 |
| 2 | Asterix | 3 | 64 | 2 |
| 3 | Asterix | 3 | 64 | 5 |

#### CF-DQN Batch 7: `scripts/cf_dqn_batch_07.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Seaquest | 1 | 32 | 2 |
| 1 | Seaquest | 1 | 32 | 5 |
| 2 | Seaquest | 1 | 64 | 2 |
| 3 | Seaquest | 1 | 64 | 5 |

#### CF-DQN Batch 8: `scripts/cf_dqn_batch_08.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Seaquest | 2 | 32 | 2 |
| 1 | Seaquest | 2 | 32 | 5 |
| 2 | Seaquest | 2 | 64 | 2 |
| 3 | Seaquest | 2 | 64 | 5 |

#### CF-DQN Batch 9: `scripts/cf_dqn_batch_09.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Seaquest | 3 | 32 | 2 |
| 1 | Seaquest | 3 | 32 | 5 |
| 2 | Seaquest | 3 | 64 | 2 |
| 3 | Seaquest | 3 | 64 | 5 |

#### CF-DQN Batch 10: `scripts/cf_dqn_batch_10.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Qbert | 1 | 32 | 2 |
| 1 | Qbert | 1 | 32 | 5 |
| 2 | Qbert | 1 | 64 | 2 |
| 3 | Qbert | 1 | 64 | 5 |

#### CF-DQN Batch 11: `scripts/cf_dqn_batch_11.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Qbert | 2 | 32 | 2 |
| 1 | Qbert | 2 | 32 | 5 |
| 2 | Qbert | 2 | 64 | 2 |
| 3 | Qbert | 2 | 64 | 5 |

#### CF-DQN Batch 12: `scripts/cf_dqn_batch_12.sh`
| GPU | Game | Seed | n_freq | max |
|-----|------|------|--------|-----|
| 0 | Qbert | 3 | 32 | 2 |
| 1 | Qbert | 3 | 32 | 5 |
| 2 | Qbert | 3 | 64 | 2 |
| 3 | Qbert | 3 | 64 | 5 |

---

## C51 Baseline Experiments (3 scripts)

### Configuration Space
- **Seeds**: 1, 2, 3
- **Games**: Breakout, Asterix, Seaquest, Qbert

### Script Organization

#### C51 Batch 1: `scripts/c51_batch_01.sh`
| GPU | Game | Seed |
|-----|------|------|
| 0 | Breakout | 1 |
| 1 | Asterix | 1 |
| 2 | Seaquest | 1 |
| 3 | Qbert | 1 |

#### C51 Batch 2: `scripts/c51_batch_02.sh`
| GPU | Game | Seed |
|-----|------|------|
| 0 | Breakout | 2 |
| 1 | Asterix | 2 |
| 2 | Seaquest | 2 |
| 3 | Qbert | 2 |

#### C51 Batch 3: `scripts/c51_batch_03.sh`
| GPU | Game | Seed |
|-----|------|------|
| 0 | Breakout | 3 |
| 1 | Asterix | 3 |
| 2 | Seaquest | 3 |
| 3 | Qbert | 3 |

---

## DQN Baseline Experiments (3 scripts)

### Configuration Space
- **Seeds**: 1, 2, 3
- **Games**: Breakout, Asterix, Seaquest, Qbert

### Script Organization

#### DQN Batch 1: `scripts/dqn_batch_01.sh`
| GPU | Game | Seed |
|-----|------|------|
| 0 | Breakout | 1 |
| 1 | Asterix | 1 |
| 2 | Seaquest | 1 |
| 3 | Qbert | 1 |

#### DQN Batch 2: `scripts/dqn_batch_02.sh`
| GPU | Game | Seed |
|-----|------|------|
| 0 | Breakout | 2 |
| 1 | Asterix | 2 |
| 2 | Seaquest | 2 |
| 3 | Qbert | 2 |

#### DQN Batch 3: `scripts/dqn_batch_03.sh`
| GPU | Game | Seed |
|-----|------|------|
| 0 | Breakout | 3 |
| 1 | Asterix | 3 |
| 2 | Seaquest | 3 |
| 3 | Qbert | 3 |

---

## Execution Summary

### Total Resources Required
- **18 SLURM jobs** (each with 4 GPUs)
- **72 GPU-jobs total** (18 scripts × 4 GPUs)
- **~9-10 hours per experiment** at 300 SPS for 10M steps
- **3 short-unkillable slots needed per experiment** (3 hours each)

### Recommended Execution Order
1. **Phase 1**: Launch all C51 and DQN baselines first (6 scripts, can complete in one wave)
2. **Phase 2**: Launch CF-DQN experiments in groups of 4-6 scripts
3. **Phase 3**: Analyze preliminary results and adjust remaining experiments if needed

### WandB Project Organization
- `CF-DQN-Breakout`: CF-DQN experiments on Breakout
- `CF-DQN-Asterix`: CF-DQN experiments on Asterix
- `CF-DQN-Seaquest`: CF-DQN experiments on Seaquest
- `CF-DQN-Qbert`: CF-DQN experiments on Qbert
- `Baselines-Breakout`: C51 and DQN on Breakout
- `Baselines-Asterix`: C51 and DQN on Asterix
- `Baselines-Seaquest`: C51 and DQN on Seaquest
- `Baselines-Qbert`: C51 and DQN on Qbert

### Monitoring
Each script will log to:
- `logs/outputs/experiment-{job_id}.{gpu_id}.out`
- `logs/errors/experiment-{job_id}.{gpu_id}.err`

Check progress with:
```bash
watch squeue -u $USER
tail -f logs/outputs/experiment-*.out
```
