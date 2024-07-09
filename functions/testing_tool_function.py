import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t
import math
from math import sqrt
from scipy.stats import norm

# Significance test function
def significance_test(control_size, control_conversion, test_size, test_conversion, confidence_interval):
    # Calculate conversion rates
    control_cvr = control_conversion / control_size
    test_cvr = test_conversion / test_size

    # Fixes the error when control and test cvr are the same 
    control_cvr = control_cvr*1.000000001 if control_cvr == test_cvr else control_cvr

    # Calculate standard deviations
    control_std = np.sqrt(control_cvr * (1 - control_cvr))
    test_std = np.sqrt(test_cvr * (1 - test_cvr))
    
    cvr_lift = test_cvr / control_cvr
    
    # Calculate pooled standard deviation
    sp = np.sqrt(((control_size - 1) * control_std ** 2 + (test_size - 1) * test_std ** 2) / (test_size + control_size - 2))

    # T-Score
    t_score = np.abs((test_cvr - control_cvr) / (sp * np.sqrt(1/test_size + 1/control_size))) 
    
    # Calculate degrees of freedom
    df = test_size + control_size - 2

    # Calculate p-value
    p_value = 2 * (1 - t.cdf(np.abs(t_score), df))
    
    # Print results
    confidence_interval = float(confidence_interval.strip('%')) / 100
    threshold = 1 - confidence_interval
    
    if p_value < threshold:
        significant_result = '<span style="color:green;"><b>Significant Test!</b></span>'
    else:
        significant_result = '<span style="color:red;"><b>Result not significant!</b></span>'

    if p_value < threshold and test_cvr > control_cvr:
        result_message = f'<b>{significant_result}</b>\n\nTest Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift * 100:.0f}% higher than Control Group conversion rate ({control_cvr*100:.4f}%).\n\n<b>You can be {int(confidence_interval*100)}% confident that Test Group will perform better than Control Group.</b>'
    elif p_value < threshold and test_cvr < control_cvr:
        result_message = f'<b>{significant_result}</b>\n\nTest Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift * 100:.0f}% lower than Control Group conversion rate ({control_cvr*100:.4f}%).\n\n<b>You can be {int(confidence_interval*100)}% confident that Control Group will perform better than Test Group.</b>'
    elif p_value > threshold and test_cvr > control_cvr:
        result_message = f'<b>{significant_result}</b>\n\nTest Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift * 100:.0f}% higher than Control Group conversion rate ({control_cvr*100:.4f}%).\n\n<b>However, you cannot be {int(confidence_interval*100)}% confident that Test Group will perform better than Control Group.</b>'
    elif p_value > threshold and test_cvr < control_cvr:
        result_message = f'<b>{significant_result}</b>\n\nTest Group conversion rate ({test_cvr*100:.4f}%) was {cvr_lift * 100:.0f}% lower than Control Group conversion rate ({control_cvr*100:.4f}%).\n\n<b>However, you cannot be {int(confidence_interval*100)}% confident that Control Group will perform better than Test Group.</b>'
        
    result_dict = {
        'Metric': ['Control Rate', 'Test Rate', 't-score', 'p-value', 'Result', 'Confidence level'],
        'Result': [f'{control_cvr*100:.3}%', f'{test_cvr*100:.3}%', t_score, p_value, significant_result, f'{confidence_interval*100}%']
    }
    
    result_df = pd.DataFrame(result_dict)

    
    return result_message, result_df

