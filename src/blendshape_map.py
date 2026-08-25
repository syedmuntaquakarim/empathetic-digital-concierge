"""
blendshape_map.py
=================
Ground-truth mapping of every tensor dim to its CTRL_expr blendshape name,
derived directly from AS_MP_Amazement1.csv column headers.

CSV cols 0,1 = Frame, Time_sec  (dropped during preprocessing)
CSV cols 2-261 → tensor dims 0-259

Emotion-specific loss weights (PILLAR 2):
  Each weight answers: "How important is this dim for distinguishing
  Joy / Anger / Grief / Neutral?"

  8.0  = PRIMARY discriminator  (model MUST get this right)
  5.0  = STRONG discriminator
  3.0  = MODERATE signal
  1.0  = default (phoneme-driven, not emotion-specific)
  0.2  = low relevance (teeth, eyelashes, pupils)
  0.05 = head pose / global transforms (suppress completely)
  0.0  = constant flag dims

Key emotion signatures:
  JOY     → mouthCornerPull, mouthCornerUp, eyeCheekRaise, browRaiseOuter
  ANGER   → browDown, jawClench, mouthCornerDepress, noseNostrilDilate
  GRIEF   → browRaiseIn (inner only!), mouthCornerDepress, mouthCornerDown
  NEUTRAL → all near resting values, minimal activation
"""

import torch

# Full name list (260 entries), index = tensor dim
BLENDSHAPE_NAMES = [
    # ── BROWS (0-7) ────────────────────────────────────────────────────────
    "browDownL",           # 0  ANGER primary: brow furrow
    "browDownR",           # 1  ANGER primary: brow furrow
    "browLateralL",        # 2  general brow shape
    "browLateralR",        # 3  general brow shape
    "browRaiseInL",        # 4  GRIEF primary: inner brow oblique raise
    "browRaiseInR",        # 5  GRIEF primary: inner brow oblique raise
    "browRaiseOuterL",     # 6  JOY secondary: outer brow raise
    "browRaiseOuterR",     # 7  JOY secondary: outer brow raise
    # ── EARS (8-9) ─────────────────────────────────────────────────────────
    "earUpL",              # 8  not emotion relevant
    "earUpR",              # 9  not emotion relevant
    # ── EYES (10-50) ───────────────────────────────────────────────────────
    "eyeBlinkL",           # 10
    "eyeBlinkR",           # 11
    "eyeLidPressL",        # 12
    "eyeLidPressR",        # 13
    "eyeWidenL",           # 14 JOY/SURPRISE: eye wide open
    "eyeWidenR",           # 15 JOY/SURPRISE: eye wide open
    "eyeSquintInnerL",     # 16 JOY: Duchenne marker (true smile squint)
    "eyeSquintInnerR",     # 17 JOY: Duchenne marker
    "eyeCheekRaiseL",      # 18 JOY primary: cheek push up into eye
    "eyeCheekRaiseR",      # 19 JOY primary: cheek push up into eye
    "eyeFaceScrunchL",     # 20 ANGER/DISGUST
    "eyeFaceScrunchR",     # 21 ANGER/DISGUST
    "eyeUpperLidUpL",      # 22
    "eyeUpperLidUpR",      # 23
    "eyeRelaxL",           # 24
    "eyeRelaxR",           # 25
    "eyeLowerLidUpL",      # 26
    "eyeLowerLidUpR",      # 27
    "eyeLowerLidDownL",    # 28
    "eyeLowerLidDownR",    # 29
    "eyeLookUpL",          # 30 gaze - not emotion
    "eyeLookUpR",          # 31
    "eyeLookDownL",        # 32
    "eyeLookDownR",        # 33
    "eyeLookLeftL",        # 34
    "eyeLookLeftR",        # 35
    "eyeLookRightL",       # 36
    "eyeLookRightR",       # 37
    "eyePupilWideL",       # 38
    "eyePupilWideR",       # 39
    "eyePupilNarrowL",     # 40
    "eyePupilNarrowR",     # 41
    "eyeParallelLookDirection", # 42
    "eyelashesUpINL",      # 43  cosmetic, low weight
    "eyelashesUpINR",      # 44
    "eyelashesUpOUTL",     # 45
    "eyelashesUpOUTR",     # 46
    "eyelashesDownINL",    # 47
    "eyelashesDownINR",    # 48
    "eyelashesDownOUTL",   # 49
    "eyelashesDownOUTR",   # 50
    # ── NOSE (51-62) ───────────────────────────────────────────────────────
    "noseWrinkleL",        # 51 ANGER/DISGUST
    "noseWrinkleR",        # 52
    "noseWrinkleUpperL",   # 53
    "noseWrinkleUpperR",   # 54
    "noseNostrilDepressL", # 55
    "noseNostrilDepressR", # 56
    "noseNostrilDilateL",  # 57 ANGER primary: nostril flare
    "noseNostrilDilateR",  # 58 ANGER primary: nostril flare
    "noseNostrilCompressL",# 59
    "noseNostrilCompressR",# 60
    "noseNasolabialDeepenL",# 61 JOY: nasolabial fold deepens with smile
    "noseNasolabialDeepenR",# 62 JOY: nasolabial fold deepens with smile
    # ── CHEEKS (63-66) ─────────────────────────────────────────────────────
    "mouthCheekSuckL",     # 63
    "mouthCheekSuckR",     # 64
    "mouthCheekBlowL",     # 65
    "mouthCheekBlowR",     # 66
    # ── MOUTH / LIPS (67-190) ──────────────────────────────────────────────
    "mouthLipsBlowL",      # 67
    "mouthLipsBlowR",      # 68
    "mouthLeft",           # 69
    "mouthRight",          # 70
    "mouthUp",             # 71
    "mouthDown",           # 72
    "mouthUpperLipRaiseL", # 73 JOY: upper lip raise with smile
    "mouthUpperLipRaiseR", # 74
    "mouthLowerLipDepressL",# 75 GRIEF: lip droop
    "mouthLowerLipDepressR",# 76 GRIEF: lip droop
    "mouthCornerPullL",    # 77 JOY PRIMARY: smile corner pull
    "mouthCornerPullR",    # 78 JOY PRIMARY: smile corner pull
    "mouthStretchL",       # 79
    "mouthStretchR",       # 80
    "mouthStretchLipsCloseL",# 81
    "mouthStretchLipsCloseR",# 82
    "mouthDimpleL",        # 83 JOY: dimple formation
    "mouthDimpleR",        # 84 JOY: dimple formation
    "mouthCornerDepressL", # 85 GRIEF/ANGER PRIMARY: corner down
    "mouthCornerDepressR", # 86 GRIEF/ANGER PRIMARY: corner down
    "mouthPressUL",        # 87 ANGER: lip press (suppressing speech)
    "mouthPressUR",        # 88
    "mouthPressDL",        # 89 ANGER: lip press
    "mouthPressDR",        # 90
    "mouthLipsPurseUL",    # 91
    "mouthLipsPurseUR",    # 92
    "mouthLipsPurseDL",    # 93
    "mouthLipsPurseDR",    # 94
    "mouthLipsTowardsUL",  # 95
    "mouthLipsTowardsUR",  # 96
    "mouthLipsTowardsDL",  # 97
    "mouthLipsTowardsDR",  # 98
    "mouthFunnelUL",       # 99
    "mouthFunnelUR",       # 100
    "mouthFunnelDL",       # 101
    "mouthFunnelDR",       # 102
    "mouthLipsTogetherUL", # 103 ANGER/NEUTRAL: lips together
    "mouthLipsTogetherUR", # 104
    "mouthLipsTogetherDL", # 105
    "mouthLipsTogetherDR", # 106
    "mouthUpperLipBiteL",  # 107
    "mouthUpperLipBiteR",  # 108
    "mouthLowerLipBiteL",  # 109
    "mouthLowerLipBiteR",  # 110
    "mouthLipsTightenUL",  # 111 ANGER: tight lips
    "mouthLipsTightenUR",  # 112
    "mouthLipsTightenDL",  # 113
    "mouthLipsTightenDR",  # 114
    "mouthLipsPressL",     # 115 ANGER: pressed lips
    "mouthLipsPressR",     # 116
    "mouthSharpCornerPullL",# 117
    "mouthSharpCornerPullR",# 118
    "mouthStickyUC",       # 119 phoneme-driven (sticky lip separation)
    "mouthStickyUINL",     # 120
    "mouthStickyUINR",     # 121
    "mouthStickyUOUTL",    # 122
    "mouthStickyUOUTR",    # 123
    "mouthStickyDC",       # 124
    "mouthStickyDINL",     # 125
    "mouthStickyDINR",     # 126
    "mouthStickyDOUTL",    # 127
    "mouthStickyDOUTR",    # 128
    "mouthLipsStickyLPh1", # 129 phoneme-driven
    "mouthLipsStickyLPh2", # 130
    "mouthLipsStickyLPh3", # 131
    "mouthLipsStickyRPh1", # 132
    "mouthLipsStickyRPh2", # 133
    "mouthLipsStickyRPh3", # 134
    "mouthLipsPushUL",     # 135
    "mouthLipsPushUR",     # 136
    "mouthLipsPushDL",     # 137
    "mouthLipsPushDR",     # 138
    "mouthLipsPullUL",     # 139
    "mouthLipsPullUR",     # 140
    "mouthLipsPullDL",     # 141
    "mouthLipsPullDR",     # 142
    "mouthLipsThinUL",     # 143
    "mouthLipsThinUR",     # 144
    "mouthLipsThinDL",     # 145
    "mouthLipsThinDR",     # 146
    "mouthLipsThickUL",    # 147
    "mouthLipsThickUR",    # 148
    "mouthLipsThickDL",    # 149
    "mouthLipsThickDR",    # 150
    "mouthLipsThinInwardUL",# 151
    "mouthLipsThinInwardUR",# 152
    "mouthLipsThinInwardDL",# 153
    "mouthLipsThinInwardDR",# 154
    "mouthLipsThickInwardUL",# 155
    "mouthLipsThickInwardUR",# 156
    "mouthLipsThickInwardDL",# 157
    "mouthLipsThickInwardDR",# 158
    "mouthCornerSharpenUL",# 159
    "mouthCornerSharpenUR",# 160
    "mouthCornerSharpenDL",# 161
    "mouthCornerSharpenDR",# 162
    "mouthCornerRounderUL",# 163
    "mouthCornerRounderUR",# 164
    "mouthCornerRounderDL",# 165
    "mouthCornerRounderDR",# 166
    "mouthUpperLipTowardsTeethL",# 167
    "mouthUpperLipTowardsTeethR",# 168
    "mouthLowerLipTowardsTeethL",# 169
    "mouthLowerLipTowardsTeethR",# 170
    "mouthUpperLipShiftLeft",# 171
    "mouthUpperLipShiftRight",# 172
    "mouthLowerLipShiftLeft",# 173
    "mouthLowerLipShiftRight",# 174
    "mouthUpperLipRollInL",# 175
    "mouthUpperLipRollInR",# 176
    "mouthUpperLipRollOutL",# 177
    "mouthUpperLipRollOutR",# 178
    "mouthLowerLipRollInL",# 179
    "mouthLowerLipRollInR",# 180
    "mouthLowerLipRollOutL",# 181
    "mouthLowerLipRollOutR",# 182
    "mouthCornerUpL",      # 183 JOY PRIMARY: corner lift
    "mouthCornerUpR",      # 184 JOY PRIMARY: corner lift
    "mouthCornerDownL",    # 185 GRIEF PRIMARY: corner droop
    "mouthCornerDownR",    # 186 GRIEF PRIMARY: corner droop
    "mouthCornerWideL",    # 187 JOY: wide smile
    "mouthCornerWideR",    # 188 JOY: wide smile
    "mouthCornerNarrowL",  # 189 ANGER: narrow/tight corners
    "mouthCornerNarrowR",  # 190 ANGER: narrow/tight corners
    # ── JAW (191-204) ──────────────────────────────────────────────────────
    "jawOpen",             # 191 phoneme-driven (primary articulation)
    "jawLeft",             # 192
    "jawRight",            # 193
    "jawFwd",              # 194
    "jawBack",             # 195
    "jawClenchL",          # 196 ANGER primary: jaw clench
    "jawClenchR",          # 197 ANGER primary: jaw clench
    "jawChinRaiseDL",      # 198 GRIEF: chin raise/dimple
    "jawChinRaiseDR",      # 199
    "jawChinRaiseUL",      # 200 GRIEF: chin raise
    "jawChinRaiseUR",      # 201
    "jawChinCompressL",    # 202
    "jawChinCompressR",    # 203
    "jawOpenExtreme",      # 204
    # ── NECK (205-218) ─────────────────────────────────────────────────────
    "neckStretchL",        # 205
    "neckStretchR",        # 206
    "neckSwallowPh1",      # 207
    "neckSwallowPh2",      # 208
    "neckSwallowPh3",      # 209
    "neckSwallowPh4",      # 210
    "neckMastoidContractL",# 211
    "neckMastoidContractR",# 212
    "neckThroatDown",      # 213
    "neckThroatUp",        # 214
    "neckDigastricDown",   # 215
    "neckDigastricUp",     # 216
    "neckThroatExhale",    # 217
    "neckThroatInhale",    # 218
    # ── TEETH (219-230) ────────────────────────────────────────────────────
    "teethUpU",            # 219
    "teethUpD",            # 220
    "teethDownU",          # 221
    "teethDownD",          # 222
    "teethLeftU",          # 223
    "teethLeftD",          # 224
    "teethRightU",         # 225
    "teethRightD",         # 226
    "teethFwdU",           # 227
    "teethFwdD",           # 228
    "teethBackU",          # 229
    "teethBackD",          # 230
    # ── TONGUE (231-250) ───────────────────────────────────────────────────
    "tongueUp",            # 231
    "tongueDown",          # 232
    "tongueLeft",          # 233
    "tongueRight",         # 234
    "tongueOut",           # 235
    "tongueIn",            # 236
    "tongueBendUp",        # 237
    "tongueBendDown",      # 238
    "tongueTwistLeft",     # 239
    "tongueTwistRight",    # 240
    "tongueTipUp",         # 241
    "tongueTipDown",       # 242
    "tongueTipLeft",       # 243
    "tongueTipRight",      # 244
    "tongueWide",          # 245
    "tongueNarrow",        # 246
    "tonguePress",         # 247
    "tongueRoll",          # 248
    "tongueThick",         # 249
    "tongueThin",          # 250
    # ── HEAD POSE (251-256) ────────────────────────────────────────────────
    "HeadYaw",             # 251 SUPPRESS: not emotion signal
    "HeadPitch",           # 252 SUPPRESS
    "HeadRoll",            # 253 SUPPRESS
    "HeadTranslationX",    # 254 SUPPRESS
    "HeadTranslationY",    # 255 SUPPRESS
    "HeadTranslationZ",    # 256 SUPPRESS
    # ── FLAGS (257-259) ────────────────────────────────────────────────────
    "HeadControlSwitch",   # 257 constant flag
    "MHFDSVersion",        # 258 constant flag
    "DisableFaceOverride", # 259 constant flag
]

assert len(BLENDSHAPE_NAMES) == 260, f"Expected 260, got {len(BLENDSHAPE_NAMES)}"

# ─────────────────────────────────────────────────────────────────────────────
# EMOTION LOSS WEIGHTS — exact per-dim values
# ─────────────────────────────────────────────────────────────────────────────

def build_loss_weights():
    """
    Returns a (260,) tensor of per-dimension loss weights.
    Based on anatomical emotion signatures for Joy/Anger/Grief/Neutral.
    """
    w = torch.ones(260)

    # ── Default groups ────────────────────────────────────────────────────
    # Eyelashes: cosmetic, not emotion relevant
    w[43:51]   = 0.2
    # Gaze / pupils: not emotion relevant in this dataset
    w[30:42]   = 0.2
    # Teeth: not visible driver of emotion
    w[219:231] = 0.2
    # Tongue: phoneme-driven, minimal emotion signal
    w[231:251] = 0.3
    # Ears: irrelevant
    w[8:10]    = 0.1
    # Neck: suppress
    w[205:219] = 0.1
    # Head pose: SUPPRESS COMPLETELY
    w[251:257] = 0.05
    # Constant flag dims: zero
    w[257:260] = 0.0

    # ── JOY signatures ────────────────────────────────────────────────────
    # mouthCornerPullL/R — primary smile
    w[77]  = 8.0; w[78]  = 8.0
    # mouthCornerUpL/R — corner lift
    w[183] = 8.0; w[184] = 8.0
    # eyeCheekRaiseL/R — apple cheek, Duchenne marker
    w[18]  = 7.0; w[19]  = 7.0
    # eyeSquintInnerL/R — true smile squint (Duchenne)
    w[16]  = 6.0; w[17]  = 6.0
    # browRaiseOuterL/R — secondary brow raise with joy
    w[6]   = 5.0; w[7]   = 5.0
    # mouthDimpleL/R
    w[83]  = 5.0; w[84]  = 5.0
    # mouthCornerWideL/R — wide smile
    w[187] = 6.0; w[188] = 6.0
    # mouthUpperLipRaiseL/R — lip raise showing teeth
    w[73]  = 5.0; w[74]  = 5.0
    # noseNasolabialDeepenL/R — nasolabial fold with smile
    w[61]  = 4.0; w[62]  = 4.0
    # eyeWidenL/R — wide eyes
    w[14]  = 4.0; w[15]  = 4.0

    # ── ANGER signatures ──────────────────────────────────────────────────
    # browDownL/R — brow furrow (corrugator)
    w[0]   = 8.0; w[1]   = 8.0
    # jawClenchL/R — jaw set / clench (masseter)
    w[196] = 7.0; w[197] = 7.0
    # noseNostrilDilateL/R — nostril flare
    w[57]  = 6.0; w[58]  = 6.0
    # noseWrinkleL/R — nose wrinkle
    w[51]  = 5.0; w[52]  = 5.0
    # eyeFaceScrunchL/R — orbital scrunch
    w[20]  = 6.0; w[21]  = 6.0
    # mouthLipsTightenUL/UR/DL/DR — tight pressed lips
    w[111] = 5.0; w[112] = 5.0; w[113] = 5.0; w[114] = 5.0
    # mouthCornerNarrowL/R — narrow tight corners
    w[189] = 5.0; w[190] = 5.0
    # mouthLipsPressL/R — pressed lips
    w[115] = 5.0; w[116] = 5.0

    # ── GRIEF signatures ──────────────────────────────────────────────────
    # browRaiseInL/R — INNER brow raise only (oblique: key grief marker)
    w[4]   = 8.0; w[5]   = 8.0
    # browLateralL/R — brow lateral (inner raise causes this)
    w[2]   = 4.0; w[3]   = 4.0
    # mouthCornerDepressL/R — corner down (opposite of joy)
    w[85]  = 8.0; w[86]  = 8.0
    # mouthCornerDownL/R — corner pulled down
    w[185] = 8.0; w[186] = 8.0
    # mouthLowerLipDepressL/R — lower lip droop
    w[75]  = 6.0; w[76]  = 6.0
    # jawChinRaiseDL/DR, jawChinRaiseUL/UR — chin dimple/raise
    w[198] = 5.0; w[199] = 5.0; w[200] = 5.0; w[201] = 5.0

    # ── SHARED / phoneme-driven mouth (moderate) ──────────────────────────
    # jawOpen — primary articulation, phoneme-driven, keep moderate
    w[191] = 2.0
    # General lip shape dims (phoneme-driven but also emotion)
    w[67:77] = 2.0   # mouthLipsBlow, mouthLeft/Right, mouthUp/Down

    return w


LOSS_WEIGHTS = build_loss_weights()


# ─────────────────────────────────────────────────────────────────────────────
# Region groupings for radar charts and evaluation
# ─────────────────────────────────────────────────────────────────────────────

REGIONS = {
    "Brows":       list(range(0, 8)),
    "Eyes":        list(range(10, 22)),        # blink, widen, squint, cheekraise
    "Nose":        list(range(51, 63)),
    "Cheeks":      list(range(63, 67)),
    "Lips/Mouth":  list(range(67, 191)),
    "Jaw":         list(range(191, 205)),
    "Neck":        list(range(205, 219)),
    "Tongue":      list(range(231, 251)),
}

# Key emotion discriminator dims for quick comparison table
EMOTION_DISCRIMINATORS = {
    "Joy":     [77, 78, 183, 184, 18, 19, 16, 17, 187, 188],
    "Anger":   [0, 1, 196, 197, 57, 58, 20, 21, 189, 190],
    "Grief":   [4, 5, 85, 86, 185, 186, 75, 76, 198, 199],
    "Neutral": [],  # all dims should be low
}


if __name__ == '__main__':
    w = LOSS_WEIGHTS
    print("Loss weight verification:")
    print(f"  mouthCornerPullL  (dim 77 , JOY)   : {w[77].item():.1f}")
    print(f"  mouthCornerUpL    (dim 183, JOY)   : {w[183].item():.1f}")
    print(f"  eyeCheekRaiseL    (dim 18 , JOY)   : {w[18].item():.1f}")
    print(f"  browDownL         (dim 0  , ANGER) : {w[0].item():.1f}")
    print(f"  jawClenchL        (dim 196, ANGER) : {w[196].item():.1f}")
    print(f"  browRaiseInL      (dim 4  , GRIEF) : {w[4].item():.1f}")
    print(f"  mouthCornerDepressL(dim 85, GRIEF) : {w[85].item():.1f}")
    print(f"  mouthCornerDownL  (dim 185, GRIEF) : {w[185].item():.1f}")
    print(f"  HeadYaw           (dim 251, HEAD)  : {w[251].item():.2f}")
    print(f"  HeadControlSwitch (dim 257, FLAG)  : {w[257].item():.1f}")
    print(f"  Total dims: {len(w)}")
    print(f"  Mean weight (active dims 0-250): {w[:251].mean().item():.3f}")
