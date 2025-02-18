from pydantic import BaseModel


class PRCurvePoint(BaseModel):
    prcn: float
    rcl: float
    f1: float
    thrd: float
