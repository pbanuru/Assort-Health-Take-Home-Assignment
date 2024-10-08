from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Tuple, Dict

@dataclass
class PatientInfo:
    # - Collect patient's name and date of birth
    name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    # - Collect insurance information
    #     - Payer name and ID
    insurance_payer: Optional[str] = None
    insurance_id: Optional[str] = None
    # - Ask if they have a referral, and to which physician
    has_referral: Optional[bool] = None
    referred_physician: Optional[str] = None
    # - Collect chief medical complaint/reason they are coming in
    chief_complaint: Optional[str] = None
    # - Collect other demographics like address
    address: Optional[str] = None
    # - Collect contact information: phone number and optionally email
    phone_number: Optional[str] = None
    email: Optional[str] = None

