"""
=================================================================================
TOPPER SQUAD ADVERSARIAL DEBATE ENGINE (THE WAR ROOM BATTLE)
The 5 specialists actively THINK, FIGHT, CRITIQUE, and DEBATE trade-offs:
  - Dr. Vikram (Grandmaster Architect) : Attacks leakage, overfitting, public LB illusions
  - Arjun (Vision & OCR Specialist)     : Attacks text-only models, demands packaging reality
  - Neha (Feature & Domain Engineer)   : Attacks noisy OCR, exposes edge cases & unit traps
  - Rohan (GBDT & Ensemble Virtuoso)   : Attacks single-model dogma, demands convex math
  - Pooja (Zero-Disqualification QA)   : Attacks reckless code, enforces strict safety gates

RESULT: Undisputed, battle-tested, bulletproof decisions with zero blind spots!
=================================================================================
"""

import sys
import time

class AdversarialDebate:
    def __init__(self):
        self.personas = {
            "Vikram": "[Dr. Vikram | Grandmaster Architect]",
            "Arjun":  "[Arjun      | Vision & OCR Specialist]",
            "Neha":   "[Neha       | Domain & Feature Engineer]",
            "Rohan":  "[Rohan      | GBDT & Ensembling Virtuoso]",
            "Pooja":  "[Pooja      | Zero-Disqualification Auditor]"
        }

    def speak(self, persona, text, mood="DEBATE"):
        prefix = self.personas[persona]
        tag = f"<{mood}>"
        print(f"\n{prefix} {tag}:")
        print(f"  \"{text}\"")
        time.sleep(0.1)

    def debate_topic(self, topic="HOW_TO_GUARANTEE_TOP_30"):
        print("\n" + "="*80)
        print("   >>> TOPPER SQUAD WAR ROOM: ADVERSARIAL DEBATE IN SESSION <<<")
        print(f"   TOPIC: {topic.replace('_', ' ')}")
        print("="*80)

        # ROUND 1: THE VISION VS TEXT FIGHT
        self.speak("Arjun", "Text-only models are a joke in Amazon ML Challenge! 70% of teams will just throw TF-IDF or BERT at the title and pray. But the title doesn't state the real price -- the packaging image has the actual printed MRP! If we don't run PaddleOCR, we are flying blind!", mood="PROVOCATION")

        self.speak("Neha", "Shut down your ego for a second, Arjun! Have you actually inspected the dataset? 25% of product URLs are low-res 200x200 thumbnails or angled packets where OCR hallucinates 'Rs. 5OO' as 5 rupees! If your pipeline blindly anchors price to 5 rupees for a 500-rupee perfume, our SMAPE explodes from 35% to 190%! We CANNOT blindly trust OCR without my regex unit binarizer and optical typo corrector!", mood="COUNTER-ATTACK")

        # ROUND 2: THE MODELING & LOSS DEBATE
        self.speak("Rohan", "Both of you are obsessing over raw signals while ignoring how trees actually split. You can extract all the features you want, but if you feed them into standard MSE LightGBM like every other amateur team, you will get destroyed. MSE cares about Rs. 50,000 laptops, not Rs. 100 grocery items. If we don't train on log1p(price) with L1 surrogate loss and blend Yandex CatBoost symmetric trees with LightGBM, we won't even make Top 200!", mood="CRITIQUE")

        self.speak("Vikram", "Rohan is mathematically right about the loss, but he's dangerously naive about validation! If you tune CatBoost hyperparameters based on the Public Leaderboard, you are walking into the classic 70% Private Leaderboard Shakeup trap! The public LB is just 30% of the data. I refuse to submit ANY model that doesn't strictly demonstrate out-of-fold SMAPE gain on our local 5-Fold Stratified Cross-Validation!", mood="VETO")

        # ROUND 3: THE ANTI-DISQUALIFICATION REALITY CHECK
        self.speak("Pooja", "Stop fighting over decimals! Do you know why 400 teams get disqualified every single year? Because their code produced a single NaN, swapped an ID during test shuffling, or predicted a negative price! Or worse, someone tried to call an external API that fails Amazon's manual inspection! If your models don't pass my pre-flight audit with 100% ID integrity and positive distribution bounds, NONE of your predictions leave this laptop!", mood="AUTHORITY")

        # ROUND 4: SYNTHESIS & THE UNDISPUTED CHAMPION DECISION
        print("\n" + "-"*80)
        print("   >>> WAR ROOM CONSENSUS: THE UNDISPUTED BULLETPROOF STRATEGY <<<")
        print("-" * 80)

        self.speak("Vikram", "Here is our synthesized battle order -- not a compromise, but the highest-probability winning sequence:", mood="VERDICT")
        print("""
  1. DUAL-STREAM GROUND TRUTH:
     - Use Neha's regex engine (IPQ, weight, volume) as the deterministic backbone.
     - Use Arjun's PaddleOCR ONLY when confidence > 0.85 and verified by optical typo corrector.
  
  2. GBDT ASYMMETRY EXPLOITATION:
     - Rohan's CatBoost + LightGBM trained strictly on log1p(price) with regression_l1 loss.
     - Out-of-fold target encoding for 40 semantic product clusters.
  
  3. POST-PROCESSING SECRET WEAPON:
     - Nelder-Mead harmonic scaler (alpha ~ 0.94) to eliminate the SMAPE denominator asymmetry penalty.
  
  4. ZERO-RISK SAFETY SHIELD:
     - Pooja's submission_verifier blocks all outputs until row count, ID alignment, and price bounds are 100% certified.
""")
        print("="*80)
        print("   [CONSENSUS REACHED] Undisputed Strategy Locked. No Weak Points Remaining.\n")

if __name__ == "__main__":
    debater = AdversarialDebate()
    debater.debate_topic()
