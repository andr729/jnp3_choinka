#include "WinMain.h"
#include "D3DApp.h"
#include <cstdlib>

LRESULT CALLBACK WindowProc(HWND hwnd, UINT Msg, WPARAM wParam, LPARAM lParam) {
	try {
		switch (Msg) {
		case WM_CREATE:
			InitDirect3D(hwnd);
			SetTimer(hwnd, 200, 10, nullptr);
			return 0;
		case WM_DESTROY:
			PostQuitMessage(0);
			return 0;
		case WM_TIMER:
			OnUpdate(hwnd);
			InvalidateRect(hwnd, nullptr, true);
			return 0;
		case WM_PAINT:
			OnRender(hwnd);
			ValidateRect(hwnd, nullptr);
			return 0;
		}
	}
	catch (...) {
		PostQuitMessage(0);
		return 0;
	}

	return DefWindowProc(hwnd, Msg, wParam, lParam);
}

INT WINAPI wWinMain(_In_ [[maybe_unused]] HINSTANCE instance,
	_In_opt_ [[maybe_unused]] HINSTANCE prev_instance,
	_In_ [[maybe_unused]] PWSTR cmd_line,
	_In_ [[maybe_unused]] INT cmd_show) {

	WNDCLASSEX wcex;
	wcex.cbSize = sizeof(WNDCLASSEX);
	wcex.style = CS_HREDRAW | CS_VREDRAW;
	wcex.lpfnWndProc = WindowProc;
	wcex.cbClsExtra = 0;
	wcex.cbWndExtra = 0;
	wcex.hInstance = instance;
	wcex.hIcon = nullptr;
	wcex.hCursor = LoadCursor(instance, IDC_ARROW);
	wcex.hbrBackground = nullptr, //static_cast<HBRUSH>(GetStockObject(WHITE_BRUSH));
		wcex.lpszMenuName = nullptr;
	wcex.lpszClassName = TEXT("Choinka Class");
	wcex.hIconSm = nullptr;

	RegisterClassEx(&wcex);

	HWND hwnd = CreateWindowEx(
		0, // Optional window styles.
		wcex.lpszClassName, // Window class
		TEXT("Choinka"), // Window text
		WS_OVERLAPPEDWINDOW, // Window style

		// Size and position
		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

		nullptr, // Parent window
		nullptr, // Menu
		wcex.hInstance, // Instance handle
		nullptr // Additional application data
	);

	if (hwnd == nullptr) {
		return 1;
	}

	ShowWindow(hwnd, cmd_show);

	MSG msg = {};
	while (BOOL rv = GetMessage(&msg, nullptr, 0, 0) != 0) {
		if (rv < 0) {
			DestroyWindow(hwnd);
			return 1;
		}
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}

	DestroyWindow(hwnd);
	return 0;
}
