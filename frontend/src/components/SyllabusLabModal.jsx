import React, { useState } from 'react';
import { 
  GraduationCap, 
  X, 
  BookOpen, 
  Compass, 
  HelpCircle, 
  CheckCircle2, 
  Cpu, 
  Network, 
  Layers, 
  Search, 
  Binary,
  Code2,
  Sparkles
} from 'lucide-react';

const SYLLABUS_DATA = [
  {
    unit: 'Unit I',
    co: 'CO1',
    hours: '05 Hours',
    title: 'Introduction to Artificial Intelligence & Soft Computing',
    icon: Cpu,
    color: '#38bdf8',
    summary: 'Foundational perspectives, present state of AI, and the paradigm shift from Hard Computing to Soft Computing.',
    topics: [
      'Artificial Intelligence (AI), AI Perspectives, Applications, Present state of AI',
      'Soft Computing Constituents: Fuzzy Logic (FL), Neuro-Computing (NN), Evolutionary Computing (EC)',
      'Characteristics of Neuro-Computing & Soft Computing: massive parallelism, fault tolerance',
      'Difference between Hard Computing and Soft Computing',
      'Concepts of Learning and Adaptation',
      'Self-learning: Recent advances in Generative AI, LLMs, and ethical challenges in AI'
    ],
    theoryCompare: {
      title: 'Hard Computing vs. Soft Computing (Lotfi A. Zadeh, 1994)',
      leftHeader: 'Hard Computing',
      leftPoints: [
        'Guiding principle: Precision, certainty, mathematical rigor',
        'Requires exact analytical mathematical models',
        'Based on binary crisp logic (0 or 1)',
        'Brittle to noise, partial truth, and imprecision',
        'High computational complexity on real-world chaotic data'
      ],
      rightHeader: 'Soft Computing',
      rightPoints: [
        'Guiding principle: Tolerance of imprecision, uncertainty, and partial truth',
        'Exploits approximate reasoning to achieve low solution cost and robustness',
        'Constituents: Fuzzy Sets + Neural Networks + Probabilistic/Genetic Search',
        'Robust to sensor noise, incomplete knowledge, and linguistic ambiguity',
        'Tractable, adaptable, and mirrors human biological cognition'
      ]
    },
    formulas: [
      {
        name: 'Soft Computing Constituency Principle',
        expr: 'SC = Fuzzy Logic ∪ Neuro-Computing ∪ Evolutionary Computing ∪ Probabilistic Reasoning',
        note: 'Synergistic combination where FL provides approximate reasoning, NN provides learning & adaptation, and EC provides global search.'
      }
    ],
    examTip: 'Midsem Viva favorite: "Why do we say ChatLab uses Soft Computing?" Answer: It couples fuzzy membership inference for prompt routing with neural representations and LLM probabilistic generation.',
    socraticPrompt: 'Explain the fundamental differences between Hard Computing and Soft Computing from first principles, and why tolerance for imprecision is essential for real-world AI systems.'
  },
  {
    unit: 'Unit II',
    co: 'CO2',
    hours: '07 Hours',
    title: 'Solving Problems by Searching',
    icon: Search,
    color: '#4ade80',
    summary: 'State space formulation, uninformed exhaustive search, heuristic A* search, local optimization, and adversarial game playing.',
    topics: [
      'Problem formulation as state space search: Initial state s₀, Actions(s), Transition model, Goal test, Path cost',
      'Uninformed Search: Depth First Search (DFS, stack, O(bm) space), Breadth First Search (BFS, queue, O(b^d) space)',
      'Informed Heuristic Search: Greedy Best-First Search (f(n) = h(n))',
      'A* Search Algorithm: f(n) = g(n) + h(n), Admissibility & Monotonicity/Consistency',
      'Local Search: Hill Climbing (plateaus/ridges), Simulated Annealing (Metropolis criterion)',
      'Self-learning: Game Playing & Adversarial Search (Minimax, Alpha-Beta pruning condition α ≥ β)'
    ],
    theoryCompare: {
      title: 'Search Strategy Comparison',
      leftHeader: 'Uninformed Search (BFS / DFS)',
      leftPoints: [
        'No domain knowledge of distance to goal',
        'BFS: Complete, optimal for uniform cost, but O(b^d) exponential RAM usage',
        'DFS: Modest linear space O(bm), but incomplete in infinite state graphs'
      ],
      rightHeader: 'Informed A* Search (Hart et al., 1968)',
      rightPoints: [
        'Combines cost-so-far g(n) with estimated remaining cost h(n): f(n) = g(n) + h(n)',
        'Guaranteed optimal if h(n) is admissible: h(n) ≤ h*(n) and h(goal) = 0',
        'No node re-expansion if h(n) is consistent (satisfies triangle inequality)',
        'Prunes provably sub-optimal search paths exponentially faster'
      ]
    },
    formulas: [
      {
        name: 'A* Evaluation Function & Admissibility',
        expr: 'f(n) = g(n) + h(n)  where  h(n) ≤ h*(n)',
        note: 'g(n) is the exact cost from root to n; h(n) is the heuristic lower-bound to goal.'
      },
      {
        name: 'Simulated Annealing Metropolis Criterion',
        expr: 'P(accept downhill move) = exp(ΔE / T)',
        note: 'Stochastically escapes local maxima by accepting worse moves with temperature-governed probability.'
      },
      {
        name: 'Alpha-Beta Pruning Invalidation Condition',
        expr: 'Prune subtree when  α ≥ β',
        note: 'α is the best value MAX can guarantee; β is the best value MIN can guarantee.'
      }
    ],
    examTip: 'Key proof question: Prove that if heuristic h(n) never overestimates the true remaining cost, A* tree search is guaranteed to return an optimal path.',
    socraticPrompt: 'Derive the mathematical proof for why the A* search algorithm is optimal when its heuristic function h(n) is admissible, and explain what happens if h(n) overestimates the true cost.'
  },
  {
    unit: 'Unit III',
    co: 'CO3',
    hours: '07 Hours',
    title: 'Knowledge and Reasoning',
    icon: Binary,
    color: '#f59e0b',
    summary: 'Symbolic representation, Propositional Logic (PL), First-Order Predicate Logic (FOPL), forward/backward chaining, and Robinson Resolution Refutation.',
    topics: [
      'Knowledge-based agents: Wumpus world, syntax and semantics of Propositional Logic',
      'Inference, Soundness (KB ⊢ α ⇒ KB ⊨ α) and Completeness (KB ⊨ α ⇒ KB ⊢ α)',
      'First-Order Predicate Logic (FOPL): objects, predicates, functions, ∀ and ∃ quantifiers',
      'Forward Chaining (data-driven) vs. Backward Chaining (goal-driven)',
      'Conjunctive Normal Form (CNF) conversion & Skolemization',
      'Robinson Resolution Refutation principle with Most General Unifier (MGU)',
      'Uncertain Knowledge: Axioms of probability, Bayes\' Rule P(H|E) = P(E|H)P(H)/P(E)'
    ],
    theoryCompare: {
      title: 'Propositional Logic vs. First-Order Predicate Logic',
      leftHeader: 'Propositional Logic (PL)',
      leftPoints: [
        'Declarative atomic propositions (P, Q)',
        'Boolean connectives (∧, ∨, ¬, ⇒, ⇔)',
        'Lacks objects, relationships, and quantifiers',
        'Combinatorial explosion when describing general world rules'
      ],
      rightHeader: 'First-Order Predicate Logic (FOPL)',
      rightPoints: [
        'Represents objects (Socrates), relations (Man(x)), functions (Mother(x))',
        'Universal quantifier ∀ and Existential quantifier ∃',
        'High expressive power for domain ontologies',
        'Semi-decidable: Entailed theorems can be proven; unentailed may loop'
      ]
    },
    formulas: [
      {
        name: 'Robinson Resolution Refutation Rule',
        expr: '(l₁ ∨ ... ∨ lₖ)  ∧  (m₁ ∨ ... ∨ mₙ)  ⊢  (Resolvent)θ   where lᵢθ = ¬mⱼθ',
        note: 'To prove KB ⊨ α, negate query (KB ∧ ¬α) in CNF and resolve until deriving the empty clause □ (NIL).'
      },
      {
        name: 'Bayes\' Rule for Uncertain Knowledge',
        expr: 'P(Hypothesis | Evidence) = [P(Evidence | Hypothesis) · P(Hypothesis)] / P(Evidence)',
        note: 'Updates prior belief P(H) to posterior belief P(H|E) using likelihood P(E|H).'
      }
    ],
    examTip: 'High-frequency midsem question: Convert an English sentence into FOPL, convert to CNF (skolemizing existential variables), and apply Resolution Refutation to prove the query.',
    socraticPrompt: 'Walk me through the 8 steps required to convert a First-Order Logic formula into Conjunctive Normal Form (CNF), including Skolemization, and demonstrate Robinson Resolution Refutation with an example.'
  },
  {
    unit: 'Unit IV',
    co: 'CO4',
    hours: '07 Hours',
    title: 'Fuzzy Set Theory & Fuzzy Inference Systems (FIS)',
    icon: Sparkles,
    color: '#ec4899',
    summary: 'Gradual membership functions, fuzzy set operations, Mamdani FIS vs. Takagi-Sugeno FIS, and centroid defuzzification.',
    topics: [
      'Crisp sets vs. Fuzzy sets: Characteristic function χ_A(x) ∈ {0,1} vs. Membership function μ_A(x) ∈ [0,1]',
      'Membership functions: Triangular (trimf), Trapezoidal (trapmf), Gaussian bell',
      'Fuzzy operations: Union (OR / max T-conorm), Intersection (AND / min T-norm), Complement (1 - μ(x))',
      'Alpha-cut: A_α = {x ∈ X | μ_A(x) ≥ α}',
      'Fuzzy Inference Systems (FIS): Fuzzification, Rule evaluation, Aggregation, Defuzzification',
      'Mamdani FIS (fuzzy consequents, Centroid defuzzification) vs. Sugeno / TSK FIS (polynomial consequents, Weighted Average)'
    ],
    theoryCompare: {
      title: 'Mamdani FIS vs. Takagi-Sugeno-Kang (TSK) FIS',
      leftHeader: 'Mamdani FIS (1975)',
      leftPoints: [
        'Consequent is a fuzzy membership set: "IF x is A THEN y is B"',
        'Aggregation uses Max or Sum of truncated output fuzzy sets',
        'Defuzzification requires numerical integration (Centroid of Area)',
        'Extremely intuitive and human-interpretable linguistic rules',
        'Implemented directly in ChatLab\'s fuzzy_router.py!'
      ],
      rightHeader: 'Takagi-Sugeno-Kang (Sugeno) FIS (1985)',
      rightPoints: [
        'Consequent is a crisp mathematical function: "IF x is A THEN z = px + qy + r"',
        'No fuzzy aggregation surface required',
        'Defuzzification is a fast Weighted Average: z* = Σ wᵢzᵢ / Σ wᵢ',
        'Computationally efficient for real-time control systems',
        'Forms the direct functional mathematical core of ANFIS!'
      ]
    },
    formulas: [
      {
        name: 'Mamdani Centroid Defuzzification (Center of Area)',
        expr: 'z* = [ ∫ z · μ_C(z) dz ] / [ ∫ μ_C(z) dz ]',
        note: 'Computes the balance point of the aggregated output fuzzy area.'
      },
      {
        name: 'Sugeno Weighted Average Defuzzification',
        expr: 'z* = [ Σ (wᵢ · zᵢ) ] / [ Σ wᵢ ]   where zᵢ = pᵢx + qᵢy + rᵢ',
        note: 'wᵢ is the rule firing strength (T-norm product/min of antecedents).'
      }
    ],
    examTip: 'Numerical alert: Practice calculating Centroid defuzzification for overlapping triangular output sets, or Sugeno weighted average for 2-rule systems.',
    socraticPrompt: 'Contrast Mamdani and Takagi-Sugeno Fuzzy Inference Systems in detail. Write out their rule formats, defuzzification equations, and explain which one is better suited for neural network hybrid learning.'
  },
  {
    unit: 'Unit V',
    co: 'CO5',
    hours: '08 Hours',
    title: 'Neural Networks: Foundations to Modern Learning',
    icon: Network,
    color: '#8b5cf6',
    summary: 'Biological vs. Artificial neurons, McCulloch-Pitts threshold model, Rosenblatt Perceptron, Minsky XOR limitation, Backpropagation, and Unsupervised SOM.',
    topics: [
      'Biological neuron structure (soma, dendrites, axon, synapses) vs. Artificial neuron',
      'McCulloch-Pitts (MCP) Neuron (1943): Excitatory inputs, Inhibitory veto inputs, Threshold logic',
      'Rosenblatt\'s Perceptron: Learning rule Δw = η(y - ŷ)x, Perceptron Convergence Theorem',
      'Linear Separability: Minsky & Papert (1969) XOR proof & the first AI Winter',
      'Multilayer Perceptron (MLP) & Error Backpropagation with Generalized Delta Rule',
      'Unsupervised Learning: Hebbian Learning ("fire together wire together"), Competitive Learning (Winner-Take-All)',
      'Kohonen Self-Organizing Maps (SOM) topological neighborhood updates & Learning Vector Quantization (LVQ)'
    ],
    theoryCompare: {
      title: 'McCulloch-Pitts vs. Learnable Perceptron',
      leftHeader: 'McCulloch-Pitts Neuron (1943)',
      leftPoints: [
        'Inputs are strictly binary {0, 1}',
        'Weights are fixed integers (+1 for excitatory, -∞ for inhibitory veto)',
        'Fixed threshold θ; cannot learn or adjust from error signals',
        'Can implement AND, OR, NOT gates'
      ],
      rightHeader: 'Rosenblatt Perceptron (1958)',
      rightPoints: [
        'Inputs and weights are continuous real numbers',
        'Learnable weights updated via error gradient: Δwᵢ = η(y - ŷ)xᵢ',
        'Guaranteed convergence on linearly separable data',
        'Fails on non-linearly separable logic functions like XOR without hidden layers'
      ]
    },
    formulas: [
      {
        name: 'Perceptron Weight Update Rule',
        expr: 'w(t+1) = w(t) + η · (y - ŷ) · x',
        note: 'Weight update is non-zero only when prediction ŷ does not match ground truth y.'
      },
      {
        name: 'Backpropagation Error Signal (Delta Rule)',
        expr: 'δ_j = (y_j - ŷ_j) · σ\'(z_j)  (output)  |  δ_j = [Σ_k δ_k w_kj] · σ\'(z_j)  (hidden)',
        note: 'Error propagates backward via multivariable chain rule through hidden layer representations.'
      },
      {
        name: 'Kohonen SOM Weight Adaptation',
        expr: 'Δw_i = η(t) · h_ci(t) · (x - w_i)   where h_ci(t) = exp(-||r_c - r_i||² / 2σ²(t))',
        note: 'Best Matching Unit (BMU) c and its spatial lattice neighbors update weights toward input x.'
      }
    ],
    examTip: 'Classic midsem proof: Show algebraically why a single-layer perceptron with weights w₁, w₂ and bias b cannot satisfy the XOR truth table.',
    socraticPrompt: 'Provide the algebraic proof for why a single-layer perceptron cannot solve the XOR problem, and explain how multi-layer architectures with backpropagation overcome this limitation.'
  },
  {
    unit: 'Unit VI',
    co: 'CO6',
    hours: '07 Hours',
    title: 'Hybrid Systems & Adaptive Neuro-Fuzzy Inference System (ANFIS)',
    icon: Layers,
    color: '#06b6d4',
    summary: 'Neuro-Fuzzy integration, Jang\'s 5-Layer ANFIS architecture, and the two-pass Hybrid Learning Algorithm (LSE + Backprop).',
    topics: [
      'Hybrid Systems: Motivation, Strengths and weaknesses of Fuzzy Logic vs. Neural Networks',
      'Neuro-Fuzzy Systems: Cooperative, Concurrent, and Integrated architectures',
      'ANFIS (J.-S. R. Jang, 1993): Functional equivalence to first-order Sugeno FIS',
      'ANFIS 5-Layer Architecture: Layer 1 (Premise), Layer 2 (Rules), Layer 3 (Normalization), Layer 4 (Consequent), Layer 5 (Summation)',
      'Hybrid Learning Algorithm: Forward pass (Least Squares Estimation for linear consequent parameters {p, q, r}) + Backward pass (Gradient Descent Backprop for non-linear premise parameters {a, b, c})',
      'Comparative Evaluation: Fuzzy vs. Neural vs. Neuro-Fuzzy systems'
    ],
    theoryCompare: {
      title: 'Neural Networks vs. Fuzzy Systems vs. ANFIS',
      leftHeader: 'Pure Neural / Pure Fuzzy',
      leftPoints: [
        'Neural Networks: High learning ability & adaptation, but black-box opaque representations',
        'Fuzzy Systems: High human interpretability & domain rules, but cannot learn or self-tune membership parameters from raw data'
      ],
      rightHeader: 'ANFIS Hybrid Neuro-Fuzzy (Jang, 1993)',
      rightPoints: [
        'Best of both worlds: Transparent IF-THEN rules + automatic gradient parameter tuning',
        'Premise parameters tune fuzzy membership curves; consequent parameters tune regression equations',
        'Rapid convergence via dual-pass Hybrid Learning (LSE + Backprop)'
      ]
    },
    anfisLayers: [
      { layer: 'Layer 1', name: 'Fuzzification (Square nodes)', formula: 'O_{1,i} = μ_{A_i}(x) = 1 / [1 + |(x - c_i)/a_i|^{2b_i}]', desc: 'Computes fuzzy membership values using bell or Gaussian premise parameters {a_i, b_i, c_i}.' },
      { layer: 'Layer 2', name: 'Rule Firing Strength (Circle nodes)', formula: 'O_{2,i} = w_i = μ_{A_i}(x) · μ_{B_i}(y)', desc: 'Computes rule firing strength via T-norm product of incoming membership grades.' },
      { layer: 'Layer 3', name: 'Normalized Firing Strength (Circle nodes)', formula: 'O_{3,i} = w̄_i = w_i / (w_1 + w_2)', desc: 'Normalizes the firing strength of rule i relative to all fired rules.' },
      { layer: 'Layer 4', name: 'Consequent Polynomials (Square nodes)', formula: 'O_{4,i} = w̄_i · f_i = w̄_i · (p_i x + q_i y + r_i)', desc: 'Multiplies normalized strength by first-order Sugeno linear equation with consequent parameters {p, q, r}.' },
      { layer: 'Layer 5', name: 'Overall Output Summation (Circle node)', formula: 'O_{5,1} = Σ w̄_i · f_i = (Σ w_i f_i) / (Σ w_i)', desc: 'Single fixed node that sums all incoming rule contributions to yield the final crisp output.' }
    ],
    formulas: [
      {
        name: 'ANFIS Hybrid Learning Two-Pass Rule',
        expr: 'Forward: LSE identifies {pᵢ, qᵢ, rᵢ}  |  Backward: Backpropagation tunes {aᵢ, bᵢ, cᵢ}',
        note: 'Consequent parameters are globally optimal via linear regression; premise parameters adapt via gradient descent.'
      }
    ],
    examTip: 'The Guaranteed 10-Mark Question: Draw the 5-layer ANFIS architecture for a 2-input, 2-rule first-order Sugeno system, state each node\'s function, and explain the Hybrid Learning algorithm.',
    socraticPrompt: 'Explain the 5-layer ANFIS architecture in detail. Describe the mathematical operation of each layer and derive how the Hybrid Learning Algorithm separates premise and consequent parameter updates.'
  }
];

export default function SyllabusLabModal({ isOpen, onClose, onAskPrompt, onOpenRoadmap, onOpenQuiz }) {
  const [selectedUnitIdx, setSelectedUnitIdx] = useState(0);

  if (!isOpen) return null;

  const currentUnit = SYLLABUS_DATA[selectedUnitIdx];
  const IconComponent = currentUnit.icon;

  return (
    <div className="modal-backdrop" onClick={onClose} style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: 'rgba(5, 8, 16, 0.85)',
      backdropFilter: 'blur(10px)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 1000,
      padding: '1.25rem'
    }}>
      <div 
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: '100%',
          maxWidth: '1050px',
          maxHeight: '92vh',
          backgroundColor: '#0f172a',
          border: '1px solid rgba(99, 102, 241, 0.35)',
          borderRadius: '18px',
          boxShadow: '0 25px 60px -15px rgba(0, 0, 0, 0.9), 0 0 35px rgba(99, 102, 241, 0.2)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}
      >
        {/* Modal Header */}
        <div style={{
          padding: '1.25rem 1.75rem',
          borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
            <div style={{
              width: '42px',
              height: '42px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #6366f1 0%, #a855f7 100%)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: '#fff',
              boxShadow: '0 0 16px rgba(99, 102, 241, 0.4)'
            }}>
              <GraduationCap style={{ width: '22px', height: '22px' }} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#f8fafc', margin: 0 }}>
                  AI & Soft Computing Syllabus Master Lab
                </h2>
                <span style={{
                  fontSize: '0.72rem',
                  fontWeight: '600',
                  padding: '2px 8px',
                  borderRadius: '9999px',
                  backgroundColor: 'rgba(99, 102, 241, 0.2)',
                  color: '#818cf8',
                  border: '1px solid rgba(99, 102, 241, 0.4)'
                }}>
                  Midsem Units I–VI / CO1–CO6
                </span>
              </div>
              <p style={{ fontSize: '0.82rem', color: '#94a3b8', margin: '2px 0 0 0' }}>
                First-principles theoretical foundations, mathematical derivations, and viva revision cards.
              </p>
            </div>
          </div>

          <button 
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#94a3b8',
              cursor: 'pointer',
              padding: '6px',
              borderRadius: '8px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s'
            }}
            onMouseEnter={(e) => { e.currentTarget.style.color = '#fff'; e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.08)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.color = '#94a3b8'; e.currentTarget.style.backgroundColor = 'transparent'; }}
          >
            <X style={{ width: '20px', height: '20px' }} />
          </button>
        </div>

        {/* Unit Selection Pills Bar */}
        <div style={{
          padding: '0.75rem 1.75rem',
          backgroundColor: 'rgba(11, 15, 25, 0.95)',
          borderBottom: '1px solid rgba(255, 255, 255, 0.06)',
          display: 'flex',
          gap: '0.6rem',
          overflowX: 'auto'
        }}>
          {SYLLABUS_DATA.map((u, idx) => {
            const isSelected = idx === selectedUnitIdx;
            return (
              <button
                key={u.unit}
                onClick={() => setSelectedUnitIdx(idx)}
                style={{
                  padding: '0.5rem 0.9rem',
                  borderRadius: '10px',
                  border: isSelected ? `1.5px solid ${u.color}` : '1px solid rgba(255, 255, 255, 0.08)',
                  backgroundColor: isSelected ? 'rgba(30, 41, 59, 0.9)' : 'rgba(15, 23, 42, 0.6)',
                  color: isSelected ? '#f8fafc' : '#94a3b8',
                  fontSize: '0.82rem',
                  fontWeight: isSelected ? '700' : '500',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.45rem',
                  whiteSpace: 'nowrap',
                  transition: 'all 0.2s',
                  boxShadow: isSelected ? `0 0 14px ${u.color}33` : 'none'
                }}
              >
                <span style={{ 
                  width: '8px', 
                  height: '8px', 
                  borderRadius: '50%', 
                  backgroundColor: u.color,
                  display: 'inline-block' 
                }} />
                <span>{u.unit} ({u.co})</span>
              </button>
            );
          })}
        </div>

        {/* Modal Body: Active Unit Details */}
        <div style={{
          padding: '1.5rem 1.75rem',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '1.25rem',
          flex: 1
        }}>
          {/* Unit Hero Banner */}
          <div style={{
            padding: '1.25rem',
            borderRadius: '14px',
            background: `linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%)`,
            border: `1px solid ${currentUnit.color}44`,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            gap: '1.5rem'
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.4rem' }}>
                <span style={{
                  fontSize: '0.75rem',
                  fontWeight: '700',
                  padding: '2px 8px',
                  borderRadius: '6px',
                  backgroundColor: `${currentUnit.color}22`,
                  color: currentUnit.color,
                  border: `1px solid ${currentUnit.color}66`
                }}>
                  {currentUnit.unit} • {currentUnit.co}
                </span>
                <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                  ⏳ {currentUnit.hours}
                </span>
              </div>
              <h3 style={{ fontSize: '1.2rem', fontWeight: '700', color: '#f8fafc', margin: '0 0 0.4rem 0' }}>
                {currentUnit.title}
              </h3>
              <p style={{ fontSize: '0.86rem', color: '#cbd5e1', margin: 0, lineHeight: '1.4' }}>
                {currentUnit.summary}
              </p>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', minWidth: '180px' }}>
              <button
                onClick={() => {
                  onClose();
                  onAskPrompt(currentUnit.socraticPrompt);
                }}
                style={{
                  padding: '0.5rem 0.85rem',
                  borderRadius: '8px',
                  background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)',
                  border: 'none',
                  color: '#fff',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  justifyContent: 'center',
                  boxShadow: '0 4px 12px rgba(99, 102, 241, 0.3)'
                }}
              >
                <Sparkles style={{ width: '14px', height: '14px' }} />
                Ask Socratic Tutor
              </button>

              <button
                onClick={() => {
                  onClose();
                  onOpenRoadmap('ai_soft_computing');
                }}
                style={{
                  padding: '0.5rem 0.85rem',
                  borderRadius: '8px',
                  backgroundColor: 'rgba(255, 255, 255, 0.06)',
                  border: '1px solid rgba(255, 255, 255, 0.12)',
                  color: '#cbd5e1',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  justifyContent: 'center',
                  transition: 'all 0.2s'
                }}
                onMouseEnter={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.12)'; }}
                onMouseLeave={(e) => { e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.06)'; }}
              >
                <Compass style={{ width: '14px', height: '14px', color: '#6366f1' }} />
                View DAG Roadmap
              </button>

              <button
                onClick={() => {
                  onClose();
                  onOpenQuiz();
                }}
                style={{
                  padding: '0.5rem 0.85rem',
                  borderRadius: '8px',
                  backgroundColor: 'rgba(16, 185, 129, 0.12)',
                  border: '1px solid rgba(16, 185, 129, 0.3)',
                  color: '#34d399',
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  justifyContent: 'center',
                  transition: 'all 0.2s'
                }}
              >
                <CheckCircle2 style={{ width: '14px', height: '14px' }} />
                Take Midsem Quiz
              </button>
            </div>
          </div>

          {/* Syllabus Topics Checklist */}
          <div style={{
            padding: '1.1rem',
            borderRadius: '12px',
            backgroundColor: 'rgba(17, 24, 39, 0.75)',
            border: '1px solid rgba(255, 255, 255, 0.08)'
          }}>
            <h4 style={{ fontSize: '0.92rem', fontWeight: '700', color: '#f8fafc', margin: '0 0 0.75rem 0', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <BookOpen style={{ width: '16px', height: '16px', color: currentUnit.color }} />
              Prescribed Syllabus Topics Breakdown
            </h4>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '0.5rem' }}>
              {currentUnit.topics.map((t, idx) => (
                <div key={idx} style={{
                  padding: '0.45rem 0.65rem',
                  borderRadius: '8px',
                  backgroundColor: 'rgba(255, 255, 255, 0.03)',
                  border: '1px solid rgba(255, 255, 255, 0.05)',
                  fontSize: '0.82rem',
                  color: '#cbd5e1',
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '0.4rem'
                }}>
                  <span style={{ color: currentUnit.color, fontWeight: '700' }}>•</span>
                  <span>{t}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Theoretical Distinction Matrix */}
          {currentUnit.theoryCompare && (
            <div style={{
              borderRadius: '12px',
              backgroundColor: 'rgba(17, 24, 39, 0.75)',
              border: '1px solid rgba(255, 255, 255, 0.08)',
              overflow: 'hidden'
            }}>
              <div style={{
                padding: '0.75rem 1.1rem',
                borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
                backgroundColor: 'rgba(30, 41, 59, 0.5)',
                fontWeight: '700',
                fontSize: '0.9rem',
                color: '#f8fafc'
              }}>
                ⚖️ {currentUnit.theoryCompare.title}
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr' }}>
                <div style={{ padding: '1rem', borderRight: '1px solid rgba(255, 255, 255, 0.08)' }}>
                  <h5 style={{ color: '#94a3b8', fontSize: '0.82rem', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
                    {currentUnit.theoryCompare.leftHeader}
                  </h5>
                  <ul style={{ paddingLeft: '1.1rem', margin: 0, fontSize: '0.82rem', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                    {currentUnit.theoryCompare.leftPoints.map((pt, i) => (
                      <li key={i}>{pt}</li>
                    ))}
                  </ul>
                </div>
                <div style={{ padding: '1rem', backgroundColor: 'rgba(99, 102, 241, 0.04)' }}>
                  <h5 style={{ color: currentUnit.color, fontSize: '0.82rem', textTransform: 'uppercase', marginBottom: '0.5rem' }}>
                    {currentUnit.theoryCompare.rightHeader}
                  </h5>
                  <ul style={{ paddingLeft: '1.1rem', margin: 0, fontSize: '0.82rem', color: '#cbd5e1', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                    {currentUnit.theoryCompare.rightPoints.map((pt, i) => (
                      <li key={i}>{pt}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* ANFIS 5-Layer Deep Dive (For Unit VI) */}
          {currentUnit.anfisLayers && (
            <div style={{
              padding: '1.1rem',
              borderRadius: '12px',
              backgroundColor: 'rgba(17, 24, 39, 0.75)',
              border: '1px solid rgba(6, 182, 212, 0.25)'
            }}>
              <h4 style={{ fontSize: '0.92rem', fontWeight: '700', color: '#38bdf8', margin: '0 0 0.75rem 0', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Layers style={{ width: '16px', height: '16px', color: '#38bdf8' }} />
                ANFIS 5-Layer Feedforward Architecture (Jang, 1993)
              </h4>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.55rem' }}>
                {currentUnit.anfisLayers.map((l, i) => (
                  <div key={i} style={{
                    padding: '0.65rem 0.85rem',
                    borderRadius: '8px',
                    backgroundColor: 'rgba(30, 41, 59, 0.5)',
                    border: '1px solid rgba(255, 255, 255, 0.06)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.25rem'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span style={{ fontWeight: '700', fontSize: '0.82rem', color: '#38bdf8' }}>{l.layer}: {l.name}</span>
                      <code style={{ fontSize: '0.78rem', backgroundColor: '#0b0f19', padding: '2px 6px', borderRadius: '4px', color: '#a5f3fc' }}>{l.formula}</code>
                    </div>
                    <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>{l.desc}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* First-Principles Mathematical Formulations */}
          <div style={{
            padding: '1.1rem',
            borderRadius: '12px',
            backgroundColor: 'rgba(17, 24, 39, 0.75)',
            border: '1px solid rgba(255, 255, 255, 0.08)'
          }}>
            <h4 style={{ fontSize: '0.92rem', fontWeight: '700', color: '#f8fafc', margin: '0 0 0.75rem 0', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Code2 style={{ width: '16px', height: '16px', color: currentUnit.color }} />
              First-Principles Mathematical Formulations
            </h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
              {currentUnit.formulas.map((fm, idx) => (
                <div key={idx} style={{
                  padding: '0.75rem 0.9rem',
                  borderRadius: '8px',
                  backgroundColor: '#0b0f19',
                  border: '1px solid rgba(99, 102, 241, 0.25)'
                }}>
                  <div style={{ fontSize: '0.78rem', fontWeight: '600', color: '#94a3b8', marginBottom: '0.25rem' }}>
                    {fm.name}
                  </div>
                  <div style={{
                    fontFamily: 'JetBrains Mono, monospace',
                    fontSize: '0.88rem',
                    color: '#a5b4fc',
                    padding: '0.25rem 0',
                    wordBreak: 'break-all'
                  }}>
                    {fm.expr}
                  </div>
                  <div style={{ fontSize: '0.76rem', color: '#64748b', marginTop: '0.25rem' }}>
                    ℹ️ {fm.note}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Exam / Viva Voce Hot Tip */}
          <div style={{
            padding: '0.9rem 1.1rem',
            borderRadius: '10px',
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem'
          }}>
            <HelpCircle style={{ width: '22px', height: '22px', color: '#f59e0b', flexShrink: 0 }} />
            <div>
              <div style={{ fontSize: '0.8rem', fontWeight: '700', color: '#fbbf24', textTransform: 'uppercase' }}>
                Midsem Examination & Viva Focus
              </div>
              <div style={{ fontSize: '0.82rem', color: '#fde68a', lineHeight: '1.35' }}>
                {currentUnit.examTip}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
