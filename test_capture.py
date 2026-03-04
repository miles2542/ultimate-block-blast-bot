import win32gui
import win32ui
import win32con
import win32api
import cv2
import numpy as np


def test_capture():
    hwnd = win32gui.FindWindow(None, "scrcpy")
    if not hwnd:
        print("scrcpy window not found by exact title. Trying substring...")

        def callback(h, hwnds):
            if "scrcpy" in win32gui.GetWindowText(h).lower():
                hwnds.append(h)
            return True

        hwnds = []
        win32gui.EnumWindows(callback, hwnds)
        if hwnds:
            hwnd = hwnds[0]
        else:
            print("scrcpy window not found at all.")
            return

    # Get client rect
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width = right - left
    height = bottom - top

    if width == 0 or height == 0:
        print("scrcpy is minimized or has 0 client area.")
        return

    # Create DCs
    hwndDC = win32gui.GetWindowDC(hwnd)
    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # Create Bitmap
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # PrintWindow - 2 is PW_RENDERFULLCONTENT (works for DWM and hardware accelerated windows usually)
    result = win32gui.SendMessage(
        hwnd,
        win32con.WM_PRINT,
        saveDC.GetSafeHdc(),
        win32con.PRF_CLIENT | win32con.PRF_CHILDREN | win32con.PRF_NONCLIENT,
    )

    # Actually wait, PrintWindow is an API:
    import ctypes

    result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)

    if result:
        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype="uint8")
        img.shape = (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4)

        # We need BGRA to BGR
        img = img[..., :3]

        cv2.imwrite("test_printwindow.png", img)
        print(
            f"Success! Captured {width}x{height} image to test_printwindow.png. Mean pixel value: {img.mean():.2f}"
        )
    else:
        print("PrintWindow failed.")

    # Cleanup
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)


if __name__ == "__main__":
    test_capture()
