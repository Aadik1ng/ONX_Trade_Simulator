import asyncio
import websockets
import json
import logging
import time
from typing import Callable, Any
import streamlit as st

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ws_client")

WS_URL = "wss://ws.gomarket-cpp.goquant.io/ws/l2-orderbook/okx/BTC-USDT-SWAP"

class OrderbookWebSocketClient:
    def __init__(self, url=WS_URL):
        self.url = url
        self.ws = None
        self.running = False
        self.on_tick: Callable[[dict, float], Any] = None  # callback(data, latency)

    async def connect(self):
        try:
            logger.info(f"Connecting to {self.url}")
            self.ws = await websockets.connect(self.url)
            self.running = True
            logger.info("WebSocket connection established.")
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.running = False

    async def listen(self):
        if not self.ws:
            await self.connect()
        while self.running:
            # Check for stop flag in Streamlit session state
            if hasattr(st.session_state, "ws_stop") and st.session_state.ws_stop:
                self.running = False
                if self.ws:
                    await self.ws.close()
                logger.info("WebSocket connection closed by stop flag.")
                break
            try:
                start_time = time.perf_counter()
                message = await self.ws.recv()
                latency = (time.perf_counter() - start_time) * 1000  # ms
                data = json.loads(message)
                logger.info(f"Received orderbook tick:\n{json.dumps(data, indent=2)}")
                if self.on_tick:
                    if asyncio.iscoroutinefunction(self.on_tick):
                        await self.on_tick(data, latency)
                    else:
                        self.on_tick(data, latency)
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed.")
                self.running = False
            except Exception as e:
                logger.error(f"Error in WebSocket listen: {e}")
                self.running = False

    async def start(self, on_tick: Callable[[dict, float], Any]):
        self.on_tick = on_tick
        await self.connect()
        await self.listen()

    async def stop(self):
        self.running = False
        if self.ws:
            await self.ws.close()
            logger.info("WebSocket connection closed.") 