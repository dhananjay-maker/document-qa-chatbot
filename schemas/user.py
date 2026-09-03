from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional

class UserCreate(BaseModel):
    name: str
    mobile_number: str
    email: Optional[EmailStr] = None
    password: str

    @field_validator("mobile_number")
    @classmethod
    def validate_mobile(cls, v):
        cleaned = v.strip()
        if not cleaned.isdigit() or len(cleaned) != 10:
            raise ValueError("Mobile number must be exactly 10 digits")
        return cleaned

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Name cannot be blank")
        return v

class UserResponse(BaseModel):
    id: int
    name: str
    mobile_number: str
    email: Optional[EmailStr] = None

    class Config:
        from_attributes = True

class UserLogin(BaseModel):
    mobile_number: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"