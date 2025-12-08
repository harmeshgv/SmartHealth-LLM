from fastapi import APIRouter
import time

router= APIRouter(prefix="/health", tags=["Health"])

start_time= time.time()



@router.get("/status")
def status():
  return {"status":"ok"}


@router.get("/ping")
def ping():
  return "pong"


@router.get("/live")
def liveness():
  return {"alive": True}


@router.get("/ready")
def readiness():
  return  {
    "ready":"yes"
  }

@router.get("/dependencies")
async def dep():
  return {
    "redis":  redis_ok()
  }

@router.get("/dignostics")
async def diagnostics():
  return {
    "uptime" : time.time() - start_time
  }