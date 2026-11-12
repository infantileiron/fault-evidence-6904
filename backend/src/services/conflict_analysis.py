"""Contradictory-evidence diagnosis: minimal conflict groups and minimum retractions.

Only a strict logZ=-inf is a conflict. A finite logZ whose exp() underflows is a
positive-probability joint evidence set and must never be reported as conflicting.
"""
import math
from scipy.special import logsumexp
from .network_validation import validate_network,validate_cpt,validate_evidence
from .log_factor import cpt_factors,product
from .information_gain import information_gain

def prepared(nodes,cpt):
 graph=validate_network(nodes);clean=graph['nodes'];rows=validate_cpt(clean,cpt)
 return clean,rows

def log_evidence_probability(nodes,cpt,evidence):
 clean,rows=prepared(nodes,cpt)
 return log_z(clean,rows,evidence)

def log_z(nodes,rows,evidence):
 """Natural-log joint probability of evidence, via the same log-factor VE as inference."""
 factors=[factor.restrict(evidence) for factor in cpt_factors(nodes,rows)]
 for variable in sorted({n['node_id'] for n in nodes}-set(evidence)):
  bucket=[factor for factor in factors if variable in factor.variables];factors=[factor for factor in factors if variable not in factor.variables]
  factors.append(product(bucket).sum_out(variable))
 return float(logsumexp(product(factors).values))

def _bits(mask):
 while mask:
  low=mask&-mask;yield low.bit_length()-1;mask^=low

def _group(items,indexes):
 return {items[i][0]:items[i][1] for i in indexes}

def _minimal_groups(items,lz,feasible,full):
 groups=[]
 for mask in range(1,full+1):
  if feasible(mask):continue
  indexes=list(_bits(mask))
  checks=[{'node':items[i][0],'value':items[i][1],'remaining_evidence':_group(items,_bits(mask^(1<<i))),'remaining_log_evidence':lz(mask^(1<<i)),'remaining_feasible':feasible(mask^(1<<i))} for i in indexes]
  if all(check['remaining_feasible'] for check in checks):
   groups.append({'observations':[{'node':items[i][0],'value':items[i][1]} for i in indexes],'size':len(indexes),'evidence':_group(items,indexes),'log_evidence':None,'log_evidence_strict_zero':True,'proper_subset_checks':checks,'basis':'联合logZ=-inf；逐项移除后logZ均有限，故为极小冲突组。'})
 return groups

def _retraction_masks(feasible,n):
 full=(1<<n)-1;best=n+1;masks=[]
 for mask in range(1<<n):
  if feasible(full^mask):
   count=mask.bit_count()
   if count<best:best=count;masks=[mask]
   elif count==best:masks.append(mask)
 return best,masks

def _plan(items,mask,target,analysis,case_candidates,newly_unobserved):
 base=analysis['baseline']
 return {'retracted_observations':[{'node':items[i][0],'value':items[i][1]} for i in _bits(mask)],'retracted_count':mask.bit_count(),'remaining_evidence':_group(items,[i for i in range(len(items)) if not mask&(1<<i)]),'log_evidence':base['log_evidence'],'evidence_probability':base['evidence_probability'],'evidence_probability_underflow':base['evidence_probability_underflow'],'target':target,'target_posterior':base['posterior'],'target_entropy':analysis['target_entropy'],'candidate_scope':analysis['eligible'],'case_candidates':case_candidates,'newly_unobserved_candidates':newly_unobserved,'candidates':[{'node_id':c['node_id'],'in_case_candidates':c['node_id'] in case_candidates,'newly_unobserved':c['node_id'] in newly_unobserved,'prior_entropy':c['prior_entropy'],'expected_entropy':c['expected_entropy'],'raw_information_gain':c['raw_information_gain'],'information_gain':c['information_gain'],'micro_negative_clipped':c['micro_negative_clipped'],'marginal_log_evidence':c['marginal']['log_evidence'],'branches':[{'state':b['state'],'probability':b['probability'],'log_probability':b['log_probability'],'probability_underflow':b['probability_underflow'],'conditional_entropy':b['conditional_entropy'],'skipped_zero_probability':b['skipped_zero_probability'],'log_evidence':b['inference']['log_evidence'] if b['inference'] else None,'target_posterior':b['inference']['posterior'] if b['inference'] else None} for b in c['branches']]} for c in analysis['candidates']]}

def analyze_conflicts(nodes,cpt,evidence,target,candidates=None):
 """Exhaustively enumerate all minimal infeasible evidence groups and all minimum retractions.

 Evidence has at most one observation per node and a network has 1-10 nodes, so the
 2^n subset sweep is exact and bounded. Feasibility is decided solely by finite logZ."""
 clean,rows=prepared(nodes,cpt);eligible=validate_evidence(clean,evidence,target,candidates)
 items=sorted(evidence.items());n=len(items);full=(1<<n)-1;cache={}
 def lz(mask):
  if mask not in cache:cache[mask]=log_z(clean,rows,_group(items,_bits(mask)))
  return cache[mask]
 def feasible(mask):return lz(mask)>-math.inf
 if feasible(full):raise ValueError('完整证据的联合logZ有限，不存在严格不可能；不执行冲突分析（exp下溢不代表冲突）')
 best,masks=_retraction_masks(feasible,n)
 plans=[]
 for mask in sorted(masks,key=lambda value:[items[i][0] for i in _bits(value)]):
  retracted=[items[i][0] for i in _bits(mask)];newly=[node for node in retracted if node!=target and node not in eligible];scope=sorted(set(eligible)|set(newly))
  analysis=information_gain(clean,cpt,_group(items,[i for i in range(n) if not mask&(1<<i)]),target,scope)
  plans.append(_plan(items,mask,target,analysis,eligible,newly))
 return {'method':'对锁定证据的全部子集做精确log因子可行性判定（logZ=-inf才冲突）','original_evidence':dict(evidence),'target':target,'log_evidence':None,'log_evidence_strict_zero':True,'minimal_conflict_groups':_minimal_groups(items,lz,feasible,full),'minimum_retraction_count':best,'retraction_plans':plans,'note':'最小冲突组是联合概率严格为0的极小证据子集；最少撤回方案只撤最少的观测并沿用原案例CPT版本重算后验与候选IG。方案仅写入本次失败底稿，不自动改写原案例或证据。'}
