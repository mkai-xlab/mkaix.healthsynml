from fastapi import APIRouter

router = APIRouter()

@router.get("")
def check_health():
    return {"status": "healthy", "message": "Knee OA API is online"}
