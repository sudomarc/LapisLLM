#!/usr/bin/env python3
"""Text generation script for LAPIS model."""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lapis.model.lapis_model import LapisModel