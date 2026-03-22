state_level_normal_prompts = [
    '{}',
    'flawless {}',
    'perfect {}',
    'unblemished {}',
    '{} without flaw',
    '{} without defect',
    '{} without damage',
]

state_level_abnormal_prompts = [
    'damaged {}',
    '{} with flaw',
    '{} with defect',
    '{} with damage',
    '{} with missing parts',
    '{} with print',
    '{} with hole',
    '{} with crack',
    '{} with scratch',
    '{} with discoloration',
    '{} with stains',
    '{} with missing parts',
    '{} with broken parts',
    '{} with bumpy surfaces',
]

state_level_normality_specific_prompts = [
    # needed to be designed category by category
]

state_level_abnormality_specific_prompts = [
    '{} with {} defect',
    '{} with {} flaw',
    '{} with {} damage',
]

template_level_prompts = [
    # 'a cropped photo of the {}',
    'a cropped photo of a {}',
    'a close-up photo of a {}',
    # 'a close-up photo of the {}',
    'a bright photo of a {}',
    # 'a bright photo of the {}',
    # 'a dark photo of the {}',
    'a dark photo of a {}',
    # 'a jpeg corrupted photo of a {}',
    # 'a jpeg corrupted photo of the {}',
    # 'a blurry photo of the {}',
    # 'a blurry photo of a {}',
    # 'a photo of a {}',
    # 'a photo of the {}',
    # 'a photo of a small {}',
    # 'a photo of the small {}',
    # 'a photo of a large {}',
    # 'a photo of the large {}',
    # 'a photo of the {} for visual inspection',
    # 'a photo of a {} for visual inspection',
    # 'a photo of the {} for anomaly detection',
    # 'a photo of a {} for anomaly detection',
]
