"""
Real-time Monitoring and Detection Pipeline for Cybersecurity.

This module implements a real-time monitoring system that continuously
analyzes network traffic, system logs, and user behavior for threat detection.
"""

import numpy as np
import pandas as pd
import threading
import queue
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import sqlite3
from collections import deque, defaultdict
import psutil
import socket
import struct
from scapy.all import sniff, IP, TCP, UDP, ICMP
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Data class for detection results."""
    timestamp: datetime
    source_ip: str
    dest_ip: str
    threat_type: str
    severity: int
    confidence: float
    features: Dict[str, Any]
    raw_data: Dict[str, Any]


@dataclass
class Alert:
    """Data class for security alerts."""
    alert_id: str
    timestamp: datetime
    severity: int
    threat_type: str
    description: str
    source_ip: str
    dest_ip: str
    confidence: float
    status: str = "active"


class DataCollector:
    """Base class for data collectors."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_running = False
        self.data_queue = queue.Queue(maxsize=1000)
        
    def start(self):
        """Start data collection."""
        self.is_running = True
        logger.info(f"Started {self.name} data collector")
        
    def stop(self):
        """Stop data collection."""
        self.is_running = False
        logger.info(f"Stopped {self.name} data collector")
        
    def collect_data(self) -> Dict[str, Any]:
        """Collect data from the source."""
        raise NotImplementedError


class NetworkTrafficCollector(DataCollector):
    """Collector for network traffic data."""
    
    def __init__(self, interface: str = None, packet_count: int = 100):
        super().__init__("NetworkTraffic")
        self.interface = interface
        self.packet_count = packet_count
        self.packet_buffer = deque(maxlen=1000)
        
    def collect_data(self) -> Dict[str, Any]:
        """Collect network traffic data."""
        try:
            # Capture packets using scapy
            packets = sniff(count=self.packet_count, iface=self.interface, timeout=1)
            
            traffic_data = []
            for packet in packets:
                if IP in packet:
                    packet_info = {
                        'timestamp': datetime.now(),
                        'src_ip': packet[IP].src,
                        'dst_ip': packet[IP].dst,
                        'protocol': packet[IP].proto,
                        'length': len(packet),
                        'flags': packet[IP].flags if hasattr(packet[IP], 'flags') else 0
                    }
                    
                    # Add protocol-specific information
                    if TCP in packet:
                        packet_info.update({
                            'src_port': packet[TCP].sport,
                            'dst_port': packet[TCP].dport,
                            'flags': packet[TCP].flags,
                            'seq': packet[TCP].seq,
                            'ack': packet[TCP].ack
                        })
                    elif UDP in packet:
                        packet_info.update({
                            'src_port': packet[UDP].sport,
                            'dst_port': packet[UDP].dport
                        })
                    elif ICMP in packet:
                        packet_info.update({
                            'type': packet[ICMP].type,
                            'code': packet[ICMP].code
                        })
                    
                    traffic_data.append(packet_info)
                    self.packet_buffer.append(packet_info)
            
            return {
                'collector': self.name,
                'timestamp': datetime.now(),
                'data': traffic_data,
                'packet_count': len(traffic_data)
            }
            
        except Exception as e:
            logger.error(f"Error collecting network traffic: {e}")
            return {'collector': self.name, 'timestamp': datetime.now(), 'data': [], 'error': str(e)}


class SystemLogCollector(DataCollector):
    """Collector for system log data."""
    
    def __init__(self, log_files: List[str] = None):
        super().__init__("SystemLogs")
        self.log_files = log_files or ['/var/log/syslog', '/var/log/auth.log']
        self.log_buffer = deque(maxlen=1000)
        
    def collect_data(self) -> Dict[str, Any]:
        """Collect system log data."""
        try:
            log_data = []
            
            for log_file in self.log_files:
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-100:]  # Read last 100 lines
                        
                        for line in lines:
                            log_entry = {
                                'timestamp': datetime.now(),
                                'source': log_file,
                                'message': line.strip(),
                                'level': self._extract_log_level(line),
                                'process': self._extract_process(line)
                            }
                            log_data.append(log_entry)
                            self.log_buffer.append(log_entry)
                            
                except FileNotFoundError:
                    logger.warning(f"Log file not found: {log_file}")
                except PermissionError:
                    logger.warning(f"Permission denied for log file: {log_file}")
            
            return {
                'collector': self.name,
                'timestamp': datetime.now(),
                'data': log_data,
                'entry_count': len(log_data)
            }
            
        except Exception as e:
            logger.error(f"Error collecting system logs: {e}")
            return {'collector': self.name, 'timestamp': datetime.now(), 'data': [], 'error': str(e)}
    
    def _extract_log_level(self, line: str) -> str:
        """Extract log level from log line."""
        levels = ['ERROR', 'WARN', 'INFO', 'DEBUG', 'CRITICAL']
        for level in levels:
            if level in line.upper():
                return level
        return 'INFO'
    
    def _extract_process(self, line: str) -> str:
        """Extract process name from log line."""
        # Simple process extraction - can be improved
        parts = line.split()
        if len(parts) > 3:
            return parts[3]
        return 'unknown'


class SystemMetricsCollector(DataCollector):
    """Collector for system performance metrics."""
    
    def __init__(self):
        super().__init__("SystemMetrics")
        self.metrics_buffer = deque(maxlen=1000)
        
    def collect_data(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            # Process metrics
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            metrics_data = {
                'timestamp': datetime.now(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv
                },
                'processes': processes[:10]  # Top 10 processes
            }
            
            self.metrics_buffer.append(metrics_data)
            
            return {
                'collector': self.name,
                'timestamp': datetime.now(),
                'data': metrics_data
            }
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {'collector': self.name, 'timestamp': datetime.now(), 'data': {}, 'error': str(e)}


class RealTimeDetector:
    """Real-time threat detector."""
    
    def __init__(self, anomaly_detector, threat_classifier=None):
        self.anomaly_detector = anomaly_detector
        self.threat_classifier = threat_classifier
        self.detection_history = deque(maxlen=1000)
        self.alert_history = deque(maxlen=1000)
        
    def detect_threats(self, data: Dict[str, Any]) -> List[DetectionResult]:
        """Detect threats in real-time data."""
        detections = []
        
        try:
            # Extract features from data
            features = self._extract_features(data)
            
            if len(features) == 0:
                return detections
            
            # Convert to numpy array for ML models
            X = np.array([features])
            
            # Anomaly detection
            if self.anomaly_detector and hasattr(self.anomaly_detector, 'predict'):
                anomaly_pred = self.anomaly_detector.predict(X)[0]
                anomaly_score = self.anomaly_detector.score_samples(X)[0] if hasattr(self.anomaly_detector, 'score_samples') else 0.0
                
                if anomaly_pred == -1:  # Anomaly detected
                    detection = DetectionResult(
                        timestamp=datetime.now(),
                        source_ip=data.get('src_ip', 'unknown'),
                        dest_ip=data.get('dst_ip', 'unknown'),
                        threat_type='Anomaly',
                        severity=self._calculate_severity(anomaly_score),
                        confidence=abs(anomaly_score),
                        features=features,
                        raw_data=data
                    )
                    detections.append(detection)
            
            # Threat classification
            if self.threat_classifier and hasattr(self.threat_classifier, 'predict'):
                threat_pred = self.threat_classifier.predict(X)[0]
                threat_proba = self.threat_classifier.predict_proba(X)[0] if hasattr(self.threat_classifier, 'predict_proba') else [0.5, 0.5]
                
                if threat_pred != 'Normal':
                    detection = DetectionResult(
                        timestamp=datetime.now(),
                        source_ip=data.get('src_ip', 'unknown'),
                        dest_ip=data.get('dst_ip', 'unknown'),
                        threat_type=threat_pred,
                        severity=self._get_threat_severity(threat_pred),
                        confidence=max(threat_proba),
                        features=features,
                        raw_data=data
                    )
                    detections.append(detection)
            
            # Store detections
            for detection in detections:
                self.detection_history.append(detection)
            
        except Exception as e:
            logger.error(f"Error in threat detection: {e}")
        
        return detections
    
    def _extract_features(self, data: Dict[str, Any]) -> List[float]:
        """Extract features from raw data."""
        features = []
        
        # Network traffic features
        if 'src_ip' in data:
            features.extend([
                len(data.get('src_ip', '')),
                len(data.get('dst_ip', '')),
                data.get('protocol', 0),
                data.get('length', 0),
                data.get('src_port', 0),
                data.get('dst_port', 0)
            ])
        
        # System metrics features
        if 'cpu' in data:
            features.extend([
                data['cpu'].get('percent', 0),
                data['memory'].get('percent', 0),
                data['disk'].get('percent', 0)
            ])
        
        # Log features
        if 'message' in data:
            features.extend([
                len(data.get('message', '')),
                data.get('level', 0) if isinstance(data.get('level'), int) else 0
            ])
        
        return features
    
    def _calculate_severity(self, anomaly_score: float) -> int:
        """Calculate threat severity from anomaly score."""
        if anomaly_score > 0.8:
            return 5  # Critical
        elif anomaly_score > 0.6:
            return 4  # High
        elif anomaly_score > 0.4:
            return 3  # Medium
        elif anomaly_score > 0.2:
            return 2  # Low
        else:
            return 1  # Info
    
    def _get_threat_severity(self, threat_type: str) -> int:
        """Get severity level for threat type."""
        severity_map = {
            'Normal': 0,
            'DoS': 4,
            'Probe': 3,
            'R2L': 5,
            'U2R': 5,
            'Anomaly': 3
        }
        return severity_map.get(threat_type, 2)


class AlertManager:
    """Manager for security alerts."""
    
    def __init__(self, alert_threshold: float = 0.7):
        self.alert_threshold = alert_threshold
        self.active_alerts = {}
        self.alert_counter = 0
        
    def process_detection(self, detection: DetectionResult) -> Optional[Alert]:
        """Process a detection and create alert if necessary."""
        if detection.confidence >= self.alert_threshold:
            alert_id = f"ALERT_{self.alert_counter:06d}"
            self.alert_counter += 1
            
            alert = Alert(
                alert_id=alert_id,
                timestamp=detection.timestamp,
                severity=detection.severity,
                threat_type=detection.threat_type,
                description=f"{detection.threat_type} detected from {detection.source_ip}",
                source_ip=detection.source_ip,
                dest_ip=detection.dest_ip,
                confidence=detection.confidence
            )
            
            self.active_alerts[alert_id] = alert
            
            logger.warning(f"Security alert generated: {alert_id} - {alert.description}")
            return alert
        
        return None
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].status = "resolved"
            logger.info(f"Alert resolved: {alert_id}")


class DatabaseManager:
    """Manager for storing detection results and alerts."""
    
    def __init__(self, db_path: str = "cybersecurity_monitoring.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create detections table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                source_ip TEXT,
                dest_ip TEXT,
                threat_type TEXT,
                severity INTEGER,
                confidence REAL,
                features TEXT,
                raw_data TEXT
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT UNIQUE,
                timestamp TEXT,
                severity INTEGER,
                threat_type TEXT,
                description TEXT,
                source_ip TEXT,
                dest_ip TEXT,
                confidence REAL,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_detection(self, detection: DetectionResult):
        """Store a detection result in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detections (timestamp, source_ip, dest_ip, threat_type, 
                                  severity, confidence, features, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            detection.timestamp.isoformat(),
            detection.source_ip,
            detection.dest_ip,
            detection.threat_type,
            detection.severity,
            detection.confidence,
            json.dumps(detection.features),
            json.dumps(detection.raw_data)
        ))
        
        conn.commit()
        conn.close()
    
    def store_alert(self, alert: Alert):
        """Store an alert in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO alerts (alert_id, timestamp, severity, threat_type,
                                          description, source_ip, dest_ip, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.alert_id,
            alert.timestamp.isoformat(),
            alert.severity,
            alert.threat_type,
            alert.description,
            alert.source_ip,
            alert.dest_ip,
            alert.confidence,
            alert.status
        ))
        
        conn.commit()
        conn.close()


class RealTimeMonitoringPipeline:
    """Main real-time monitoring pipeline."""
    
    def __init__(self, anomaly_detector=None, threat_classifier=None):
        self.anomaly_detector = anomaly_detector
        self.threat_classifier = threat_classifier
        self.detector = RealTimeDetector(anomaly_detector, threat_classifier)
        self.alert_manager = AlertManager()
        self.db_manager = DatabaseManager()
        
        # Data collectors
        self.collectors = {
            'network': NetworkTrafficCollector(),
            'system_logs': SystemLogCollector(),
            'system_metrics': SystemMetricsCollector()
        }
        
        # Pipeline control
        self.is_running = False
        self.collection_threads = {}
        self.processing_thread = None
        
    def start_monitoring(self, update_interval: float = 1.0):
        """Start the real-time monitoring pipeline."""
        logger.info("Starting real-time monitoring pipeline...")
        
        self.is_running = True
        
        # Start data collectors
        for name, collector in self.collectors.items():
            collector.start()
            thread = threading.Thread(target=self._collect_data_loop, args=(collector, update_interval))
            thread.daemon = True
            thread.start()
            self.collection_threads[name] = thread
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Real-time monitoring pipeline started")
    
    def stop_monitoring(self):
        """Stop the real-time monitoring pipeline."""
        logger.info("Stopping real-time monitoring pipeline...")
        
        self.is_running = False
        
        # Stop collectors
        for collector in self.collectors.values():
            collector.stop()
        
        # Wait for threads to finish
        for thread in self.collection_threads.values():
            thread.join(timeout=5)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("Real-time monitoring pipeline stopped")
    
    def _collect_data_loop(self, collector: DataCollector, interval: float):
        """Data collection loop for a collector."""
        while self.is_running:
            try:
                data = collector.collect_data()
                collector.data_queue.put(data, timeout=1)
            except queue.Full:
                logger.warning(f"Data queue full for {collector.name}")
            except Exception as e:
                logger.error(f"Error in data collection loop for {collector.name}: {e}")
            
            time.sleep(interval)
    
    def _processing_loop(self):
        """Main processing loop for threat detection."""
        while self.is_running:
            try:
                # Process data from all collectors
                for collector in self.collectors.values():
                    try:
                        while not collector.data_queue.empty():
                            data = collector.data_queue.get_nowait()
                            self._process_data(data)
                    except queue.Empty:
                        pass
                    except Exception as e:
                        logger.error(f"Error processing data from {collector.name}: {e}")
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(1)
    
    def _process_data(self, data: Dict[str, Any]):
        """Process collected data for threat detection."""
        try:
            # Detect threats
            detections = self.detector.detect_threats(data)
            
            # Process each detection
            for detection in detections:
                # Store detection in database
                self.db_manager.store_detection(detection)
                
                # Generate alert if necessary
                alert = self.alert_manager.process_detection(detection)
                if alert:
                    self.db_manager.store_alert(alert)
                    
                    # Log high-severity alerts
                    if alert.severity >= 4:
                        logger.critical(f"HIGH SEVERITY ALERT: {alert.description}")
                    elif alert.severity >= 3:
                        logger.warning(f"MEDIUM SEVERITY ALERT: {alert.description}")
        
        except Exception as e:
            logger.error(f"Error processing data: {e}")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        stats = {
            'is_running': self.is_running,
            'total_detections': len(self.detector.detection_history),
            'active_alerts': len(self.alert_manager.active_alerts),
            'collectors_status': {}
        }
        
        for name, collector in self.collectors.items():
            stats['collectors_status'][name] = {
                'is_running': collector.is_running,
                'queue_size': collector.data_queue.qsize()
            }
        
        return stats
    
    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts."""
        alerts = self.alert_manager.get_active_alerts()
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        return alerts[:limit]


def main():
    """Example usage of the real-time monitoring pipeline."""
    from anomaly_detection.anomaly_detection import IsolationForestDetector
    from models.ml_models import RandomForestModel
    
    # Create simple detectors for demonstration
    anomaly_detector = IsolationForestDetector()
    threat_classifier = RandomForestModel()
    
    # Generate some training data
    np.random.seed(42)
    X_normal = np.random.randn(1000, 10)
    y_mixed = np.random.randint(0, 5, 1000)
    
    # Train detectors
    anomaly_detector.fit(X_normal)
    threat_classifier.train(X_normal, y_mixed)
    
    # Create monitoring pipeline
    pipeline = RealTimeMonitoringPipeline(anomaly_detector, threat_classifier)
    
    try:
        # Start monitoring
        pipeline.start_monitoring(update_interval=2.0)
        
        # Monitor for 30 seconds
        print("Monitoring for 30 seconds...")
        time.sleep(30)
        
        # Get statistics
        stats = pipeline.get_monitoring_stats()
        print(f"Monitoring stats: {stats}")
        
        # Get recent alerts
        alerts = pipeline.get_recent_alerts()
        print(f"Recent alerts: {len(alerts)}")
        for alert in alerts:
            print(f"  - {alert.alert_id}: {alert.description}")
    
    finally:
        # Stop monitoring
        pipeline.stop_monitoring()
        print("Monitoring stopped")


if __name__ == "__main__":
    main()
