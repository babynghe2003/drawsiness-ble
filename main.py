# Import các thư viện cần thiết
import logging
from bluezero import async_tools, adapter, peripheral
from drowsiness import DrowsinessDetector

# UUID cho dịch vụ và đặc điểm BLE
CPU_TMP_SRVC = '1111'
CPU_TMP_CHRC = '2222'

list_value = [0x02, 0x03, 0x01, 0x05]

drowsiness_detector = DrowsinessDetector()

def on_connect(device_address):
    print(f"Connected from {device_address}")
    drowsiness_detector.start()


def on_disconnect(device_address):
    print("Disconnected from " + str(device_address))
    drowsiness_detector.stop()


def read_value():
    with drowsiness_detector.thread_lock:
        if drowsiness_detector.isSleeping():
            return [0x01]
        else:
            return [0x00]

def notify_callback(notifying, characteristic):
    print("Notify callback called")
    if notifying:
        async_tools.add_timer_ms(50, send_notification, characteristic)


def send_notification(characteristic):
    print("Sending notify")
    value = read_value()  # Giá trị để gửi đến điện thoại
    characteristic.set_value(value)
    return characteristic.is_notifying

# Hàm chính
def main(adapter_address):
    # Thiết lập ghi log
    logger = logging.getLogger('localGATT')
    logger.setLevel(logging.DEBUG)

    # Tạo đối tượng peripheral
    peripheral_device = peripheral.Peripheral(
        adapter_address,
        local_name='AEye Provip 3',
        appearance=1344
    )

    # Thêm dịch vụ
    peripheral_device.add_service(
        srv_id=1,
        uuid=CPU_TMP_SRVC,
        primary=True
    )

    # Thêm đặc điểm BLE với callback và notify
    peripheral_device.add_characteristic(
        srv_id=1,
        chr_id=1,
        uuid=CPU_TMP_CHRC,
        value=[],
        notifying=False,  # Bật chế độ notify
        flags=['notify'],
        notify_callback=notify_callback
    )

    peripheral_device.add_descriptor(srv_id=1, chr_id=1, dsc_id=1, uuid='2904',
                              value=[0x0E, 0xFE, 0x2F, 0x27, 0x01, 0x00,
                                     0x00],
                              flags=['read'])

    peripheral_device.on_connect = on_connect
    peripheral_device.on_disconnect = on_disconnect

    peripheral_device.publish()

if __name__ == '__main__':
    main(list(adapter.Adapter.available())[0].address)

