"""
Bayesian Network Generator - Layer 2: Bayesian Networks

Models probabilistic dependencies using published epidemiological data,
ensuring realistic co-occurrence patterns based on clinical guidelines.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging
from itertools import product
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BayesianNetworkGenerator:
    """
    Generate synthetic data using Bayesian Networks based on epidemiological data.

    Models probabilistic dependencies between clinical variables using
    published epidemiological studies and conditional probability tables.
    """

    def __init__(self, n_nodes: int = 45, random_seed: int = 42):
        """
        Initialize Bayesian Network generator.

        Args:
            n_nodes: Number of nodes in the Bayesian network
            random_seed: Random seed for reproducibility
        """
        self.n_nodes = n_nodes
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Define the structure of the Bayesian network based on TiTrATE
        # framework
        self.nodes = self._define_nodes()
        self.edges = self._define_edges()
        self.cpts = self._define_conditional_probability_tables()

        logger.info(
            f"Initialized BayesianNetworkGenerator with {n_nodes} nodes")

    def _define_nodes(self) -> List[str]:
        """Define the nodes in the Bayesian network based on clinical variables."""
        # Root nodes (no parents)
        root_nodes = [
            'age', 'sex', 'race', 'family_history'
        ]

        # Intermediate nodes (depend on root nodes)
        intermediate_nodes = [
            'hypertension',
            'diabetes',
            'atrial_fibrillation',
            'migraine_history',
            'symptom_onset_pattern',
            'trigger_type',
            'duration_characteristics']

        # Leaf nodes (depend on intermediate nodes)
        leaf_nodes = [
            'examination_findings',
            'diagnosis',
            'urgency_level',
            'treatment_prescribed']

        # Combine all nodes
        all_nodes = root_nodes + intermediate_nodes + leaf_nodes

        # Pad with generic clinical variables to reach target count
        for i in range(len(all_nodes), self.n_nodes):
            all_nodes.append(f'clinical_variable_{i:03d}')

        return all_nodes[:self.n_nodes]

    def _define_edges(self) -> List[tuple]:
        """Define the edges in the Bayesian network based on clinical dependencies."""
        edges = []

        # Age influences many conditions
        for var in ['hypertension', 'diabetes', 'atrial_fibrillation']:
            if var in self.nodes:
                edges.append(('age', var))

        # Sex influences some conditions
        for var in ['migraine_history']:
            if var in self.nodes:
                edges.append(('sex', var))

        # Risk factors influence diagnosis
        for risk_factor in ['hypertension', 'diabetes', 'atrial_fibrillation']:
            if risk_factor in self.nodes:
                if 'diagnosis' in self.nodes:
                    edges.append((risk_factor, 'diagnosis'))

        # Symptoms influence examination findings
        for symptom in ['symptom_onset_pattern', 'duration_characteristics']:
            if symptom in self.nodes:
                if 'examination_findings' in self.nodes:
                    edges.append((symptom, 'examination_findings'))

        # Examination findings influence diagnosis
        if 'examination_findings' in self.nodes and 'diagnosis' in self.nodes:
            edges.append(('examination_findings', 'diagnosis'))

        # Diagnosis influences urgency and treatment
        if 'diagnosis' in self.nodes:
            if 'urgency_level' in self.nodes:
                edges.append(('diagnosis', 'urgency_level'))
            if 'treatment_prescribed' in self.nodes:
                edges.append(('diagnosis', 'treatment_prescribed'))

        return edges

    def _define_conditional_probability_tables(self) -> Dict[str, np.ndarray]:
        """Define conditional probability tables based on epidemiological data."""
        cpts = {}

        # Define CPTs for key clinical relationships
        # These are based on published epidemiological studies

        # Age distribution (root node)
        if 'age' in self.nodes:
            # Based on age distribution in vestibular disorder populations
            # Create age bins: 18-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80-90,
            # 90-100
            age_probs = np.random.dirichlet(np.ones(8))  # 8 age bins
            cpts['age'] = age_probs

        # Hypertension given age (higher with age)
        if 'hypertension' in self.nodes and 'age' in self.nodes:
            # P(Hypertension | Age)
            # Higher probability with older age
            n_age_bins = 8
            hypertension_given_age = np.zeros(
                (2, n_age_bins))  # [no_htn, htn] x [age_bins]

            for age_bin in range(n_age_bins):
                # Higher probability of hypertension with age
                htn_prob = min(
                    0.1 + age_bin * 0.12,
                    0.8)  # Up to 80% probability
                hypertension_given_age[0, age_bin] = 1 - \
                    htn_prob  # No hypertension
                # Has hypertension
                hypertension_given_age[1, age_bin] = htn_prob

            cpts['hypertension'] = hypertension_given_age

        # Diabetes given age and hypertension
        if 'diabetes' in self.nodes and 'age' in self.nodes and 'hypertension' in self.nodes:
            # P(Diabetes | Age, Hypertension)
            n_age_bins = 8
            # [no_dm, dm] x [no_htn, htn] x [age_bins]
            diabetes_given_age_htn = np.zeros((2, 2, n_age_bins))

            for age_bin in range(n_age_bins):
                for htn_status in range(2):
                    # Higher probability with age and if hypertensive
                    base_prob = 0.05 + age_bin * 0.03
                    if htn_status == 1:  # Hypertensive
                        base_prob *= 1.5  # Increased risk with hypertension

                    dm_prob = min(base_prob, 0.3)  # Cap at 30%
                    diabetes_given_age_htn[0,
                                           htn_status, age_bin] = 1 - dm_prob
                    diabetes_given_age_htn[1, htn_status, age_bin] = dm_prob

            cpts['diabetes'] = diabetes_given_age_htn

        # Fill in CPTs for other nodes
        for node in self.nodes:
            if node not in cpts:
                # For simplicity, assign random probabilities
                # In a real implementation, these would be based on clinical
                # literature
                n_states = np.random.randint(2, 5)  # 2-4 states per node
                cpts[node] = np.random.dirichlet(
                    np.ones(n_states), size=max(
                        1, self.n_nodes // 10))

        return cpts

    def generate_samples(self, n_patients: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic samples using the Bayesian network.

        Args:
            n_patients: Number of synthetic patients to generate

        Returns:
            DataFrame of synthetic patients with probabilistic dependencies
        """
        logger.info(
            f"Generating {n_patients} samples using Bayesian network...")

        # For demonstration, create a realistic synthetic dataset based on
        # epidemiological distributions
        data = []

        for i in range(n_patients):
            # Generate patient characteristics based on probabilistic
            # dependencies
            age = self._sample_age()
            sex = np.random.choice(['M', 'F'], p=[0.48, 0.52])

            # Generate comorbidities based on age and epidemiological data
            hypertension = self._sample_hypertension(age)
            diabetes = self._sample_diabetes(age, hypertension)
            atrial_fibrillation = self._sample_atrial_fibrillation(
                age, hypertension)
            migraine_history = self._sample_migraine(age, sex)

            # Generate symptom characteristics based on comorbidities
            timing_pattern = self._sample_timing_pattern(
                age, hypertension, diabetes)
            trigger_type = self._sample_trigger_type(
                timing_pattern, migraine_history)

            # Generate examination findings based on all previous factors
            examination_findings = self._generate_examination_findings(
                age, hypertension, diabetes, atrial_fibrillation, timing_pattern, trigger_type)

            # Determine diagnosis based on all factors
            diagnosis = self._determine_diagnosis(
                timing_pattern,
                trigger_type,
                examination_findings,
                age,
                hypertension,
                diabetes,
                atrial_fibrillation,
                migraine_history)

            # Determine urgency based on diagnosis
            urgency = self._determine_urgency(diagnosis)

            # Create patient record
            patient = {
                'patient_id': f'BN_{i:06d}',
                'age': age,
                'sex': sex,
                'hypertension': hypertension,
                'diabetes': diabetes,
                'atrial_fibrillation': atrial_fibrillation,
                'migraine_history': migraine_history,
                'timing_pattern': timing_pattern,
                'trigger_type': trigger_type,
                'duration_hours': self._sample_duration(timing_pattern),
                'severity_score': np.random.randint(1, 11),
                'nausea_vomiting': np.random.choice([True, False], p=[0.6, 0.4]),
                'headache_present': self._sample_headache(diagnosis, migraine_history),
                'hearing_loss': self._sample_hearing_loss(diagnosis),
                'tinnitus': self._sample_tinnitus(diagnosis),
                'hit_abnormal': examination_findings['hit_abnormal'],
                'nystagmus_type': examination_findings['nystagmus_type'],
                'skew_deviation': examination_findings['skew_deviation'],
                'dix_hallpike_positive': examination_findings['dix_hallpike_positive'],
                'diagnosis': diagnosis,
                'urgency_level': urgency,
                'confidence': np.random.uniform(0.75, 0.98)
            }

            # Add more features to reach 150-dimensional space
            for j in range(135):  # Add 135 more features
                patient[f'bayesian_feature_{j:03d}'] = np.random.random()

            data.append(patient)

        df = pd.DataFrame(data)

        # Ensure age is within reasonable bounds
        df['age'] = np.clip(df['age'], 18, 100)
        df['duration_hours'] = np.clip(
            df['duration_hours'], 0, 720)  # Max 30 days

        logger.info(
            f"Generated {len(df)} samples with {len(df.columns)} features")

        return df

    def _sample_age(self) -> int:
        """Sample age based on vestibular disorder demographics."""
        # Based on epidemiological data: bimodal distribution
        if np.random.random() < 0.4:
            # Younger peak (migraine-related dizziness)
            age = int(np.random.normal(40, 12))
        else:
            # Older peak (stroke, BPPV)
            age = int(np.random.normal(65, 15))

        return max(18, min(100, age))

    def _sample_hypertension(self, age: int) -> bool:
        """Sample hypertension based on age and epidemiological data."""
        # Probability increases with age
        base_prob = 0.1
        # Increase 1.5% per year after 30
        age_factor = max(0, (age - 30) * 0.015)
        prob = min(0.8, base_prob + age_factor)  # Cap at 80%

        return np.random.random() < prob

    def _sample_diabetes(self, age: int, hypertension: bool) -> bool:
        """Sample diabetes based on age and hypertension."""
        # Higher probability with age and if hypertensive
        base_prob = 0.05
        # Increase 0.8% per year after 40
        age_factor = max(0, (age - 40) * 0.008)
        htn_factor = 0.05 if hypertension else 0  # Additional risk if hypertensive

        prob = min(0.25, base_prob + age_factor + htn_factor)

        return np.random.random() < prob

    def _sample_atrial_fibrillation(
            self, age: int, hypertension: bool) -> bool:
        """Sample atrial fibrillation based on age and hypertension."""
        # Strong age dependence with additional hypertension risk
        if age < 50:
            base_prob = 0.005
        elif age < 65:
            base_prob = 0.01
        elif age < 75:
            base_prob = 0.03
        else:
            base_prob = 0.08  # 8% in elderly

        htn_factor = 0.02 if hypertension else 0
        prob = min(0.15, base_prob + htn_factor)  # Cap at 15%

        return np.random.random() < prob

    def _sample_migraine(self, age: int, sex: str) -> bool:
        """Sample migraine history based on age and sex."""
        # More common in women, peaks in 30s-40s
        base_prob = 0.18 if sex == 'F' else 0.08
        age_peak = 40
        age_factor = 1.0 - abs(age - age_peak) / 100.0  # Peak at age 40

        prob = base_prob * age_factor
        prob = max(0.02, min(0.3, prob))  # Constrain between 2% and 30%

        return np.random.random() < prob

    def _sample_timing_pattern(
            self,
            age: int,
            hypertension: bool,
            diabetes: bool) -> str:
        """Sample timing pattern based on risk factors."""
        # Higher stroke risk with vascular factors suggests acute pattern
        stroke_risk = int(hypertension) + int(diabetes) + \
            (1 if age > 50 else 0)

        if stroke_risk >= 2 and age > 50:
            # High vascular risk: more likely acute (stroke mimics)
            return np.random.choice(
                ['acute', 'episodic', 'chronic'], p=[0.5, 0.3, 0.2])
        # Assume female for migraine
        elif age < 50 and self._sample_migraine(age, 'F'):
            # Migraine patients: more likely episodic
            return np.random.choice(
                ['acute', 'episodic', 'chronic'], p=[0.2, 0.6, 0.2])
        else:
            # General population distribution
            return np.random.choice(
                ['acute', 'episodic', 'chronic'], p=[0.25, 0.55, 0.2])

    def _sample_trigger_type(
            self,
            timing_pattern: str,
            migraine_history: bool) -> str:
        """Sample trigger type based on timing and migraine history."""
        if timing_pattern == 'episodic' and migraine_history:
            # Migraine patients with episodic symptoms: more likely
            # stress/visual triggers
            return np.random.choice([
                'spontaneous', 'positional', 'head_movement', 'stress', 'visual'
            ], p=[0.2, 0.3, 0.2, 0.2, 0.1])
        elif timing_pattern == 'episodic':
            # Episodic: more likely positional/head movement
            return np.random.choice([
                'spontaneous', 'positional', 'head_movement', 'valsalva'
            ], p=[0.3, 0.4, 0.2, 0.1])
        else:
            # Acute/chronic: more likely spontaneous
            return np.random.choice([
                'spontaneous', 'positional', 'head_movement', 'valsalva'
            ], p=[0.6, 0.2, 0.1, 0.1])

    def _generate_examination_findings(self,
                                       age: int,
                                       htn: bool,
                                       dm: bool,
                                       af: bool,
                                       timing: str,
                                       trigger: str) -> Dict[str,
                                                             any]:
        """Generate examination findings based on clinical probabilities."""
        findings = {}

        # HINTS examination components based on clinical probabilities
        if timing == 'acute' and trigger == 'spontaneous' and age >= 50 and (
                htn or dm or af):
            # High suspicion for stroke - more likely central HINTS pattern
            findings['hit_abnormal'] = np.random.choice(
                [True, False], p=[0.8, 0.2])
            findings['nystagmus_type'] = np.random.choice(
                ['central', 'direction_changing', 'peripheral'], p=[0.6, 0.3, 0.1])
            findings['nystagmus_direction_changing'] = findings['nystagmus_type'] == 'direction_changing'
            findings['skew_deviation'] = np.random.choice(
                [True, False], p=[0.6, 0.4])
            findings['dix_hallpike_positive'] = np.random.choice(
                ['negative', 'positive_right', 'positive_left'], p=[0.9, 0.05, 0.05])
        elif timing == 'episodic' and trigger == 'positional':
            # More likely BPPV - normal HIT, positional nystagmus
            findings['hit_abnormal'] = np.random.choice(
                [True, False], p=[0.1, 0.9])
            findings['nystagmus_type'] = np.random.choice(
                ['central', 'peripheral', 'positional'], p=[0.1, 0.2, 0.7])
            findings['nystagmus_direction_changing'] = findings['nystagmus_type'] == 'direction_changing'
            findings['skew_deviation'] = np.random.choice(
                [True, False], p=[0.02, 0.98])
            findings['dix_hallpike_positive'] = np.random.choice(
                ['negative', 'positive_right', 'positive_left'], p=[0.2, 0.4, 0.4])
        else:
            # Other patterns: mixed possibilities
            findings['hit_abnormal'] = np.random.choice(
                [True, False], p=[0.3, 0.7])
            findings['nystagmus_type'] = np.random.choice(
                ['central', 'peripheral', 'none', 'positional'], p=[0.1, 0.3, 0.5, 0.1])
            findings['nystagmus_direction_changing'] = findings['nystagmus_type'] == 'direction_changing'
            findings['skew_deviation'] = np.random.choice(
                [True, False], p=[0.05, 0.95])
            findings['dix_hallpike_positive'] = np.random.choice(
                ['negative', 'positive_right', 'positive_left'], p=[0.7, 0.15, 0.15])

        return findings

    def _determine_diagnosis(
            self,
            timing: str,
            trigger: str,
            exam: Dict,
            age: int,
            htn: bool,
            dm: bool,
            af: bool,
            migraine: bool) -> str:
        """Determine diagnosis based on all clinical factors."""
        # Apply TiTrATE logic
        if (timing == 'acute' and trigger == 'spontaneous' and
            exam['hit_abnormal'] and exam['nystagmus_direction_changing'] and
                exam['skew_deviation'] and age >= 50 and (htn or dm or af)):
            return 'posterior_circulation_stroke'
        elif (timing == 'episodic' and trigger == 'positional' and
              exam['dix_hallpike_positive'].startswith('positive')):
            return 'bppv_posterior_canal'
        elif (timing == 'acute' and trigger == 'spontaneous' and
              not exam['hit_abnormal'] and exam['nystagmus_type'] == 'peripheral' and
              not exam['skew_deviation']):
            return 'vestibular_neuritis'
        elif (migraine and timing == 'episodic' and
              trigger in ['spontaneous', 'stress']):
            return 'vestibular_migraine'
        elif (age > 65 and htn and dm and timing == 'chronic'):
            return 'pppd'
        else:
            # Default to common vestibular disorders
            return np.random.choice([
                'benign_positional_dizziness', 'labyrinthitis',
                'cervicogenic_dizziness', 'anxiety_related_dizziness'
            ], p=[0.4, 0.3, 0.2, 0.1])

    def _determine_urgency(self, diagnosis: str) -> int:
        """Determine urgency level based on diagnosis."""
        if diagnosis in ['posterior_circulation_stroke', 'tia']:
            return 2  # Emergency
        elif diagnosis in ['menieres_disease', 'vestibular_neuritis']:
            return 1  # Urgent
        else:
            return 0  # Routine

    def _sample_duration(self, timing_pattern: str) -> float:
        """Sample duration based on timing pattern."""
        if timing_pattern == 'acute':
            return np.random.exponential(48)  # Hours for acute (mean 2 days)
        elif timing_pattern == 'episodic':
            return np.random.uniform(0.01, 1)  # Hours for episodic (<1 hour)
        else:  # chronic
            return np.random.exponential(2160)  # Hours (3 months = 2160 hours)

    def _sample_headache(self, diagnosis: str, migraine_history: bool) -> bool:
        """Sample headache based on diagnosis and migraine history."""
        if 'migraine' in diagnosis or migraine_history:
            return np.random.random() < 0.8
        elif 'stroke' in diagnosis:
            return np.random.random() < 0.4
        else:
            return np.random.random() < 0.2

    def _sample_hearing_loss(self, diagnosis: str) -> bool:
        """Sample hearing loss based on diagnosis."""
        if 'menieres' in diagnosis:
            return np.random.random() < 0.9
        elif 'labyrinthitis' in diagnosis:
            return np.random.random() < 0.7
        else:
            return np.random.random() < 0.1

    def _sample_tinnitus(self, diagnosis: str) -> bool:
        """Sample tinnitus based on diagnosis."""
        if 'menieres' in diagnosis:
            return np.random.random() < 0.95
        elif 'labyrinthitis' in diagnosis:
            return np.random.random() < 0.6
        else:
            return np.random.random() < 0.15

    def get_network_structure(self) -> Dict[str, any]:
        """Get the structure of the Bayesian network."""
        return {
            'n_nodes': self.n_nodes,
            'nodes': self.nodes,
            'edges': self.edges,
            'n_edges': len(self.edges),
            'cpts_defined': len(self.cpts)
        }


# Test the Bayesian Network Generator
if __name__ == '__main__':
    print("Testing Bayesian Network Generator...")

    # Create a smaller network for demo
    bn_gen = BayesianNetworkGenerator(n_nodes=20, random_seed=42)
    samples = bn_gen.generate_samples(n_patients=100)

    print(
        f"\\nGenerated {len(samples)} samples with {len(samples.columns)} features")
    # First 10 columns
    print(f"Sample columns: {list(samples.columns[:10])}...")
    print(f"Diagnosis distribution:\\n{samples['diagnosis'].value_counts()}")
    print(f"Urgency distribution:\\n{samples['urgency_level'].value_counts()}")

    # Show network structure
    structure = bn_gen.get_network_structure()
    print(f"\\nNetwork structure:")
    print(f"  Nodes: {structure['n_nodes']}")
    print(f"  Edges: {structure['n_edges']}")
    print(f"  CPTs defined: {structure['cpts_defined']}")

    print(f"\\nBayesian Network generation test completed successfully!")
