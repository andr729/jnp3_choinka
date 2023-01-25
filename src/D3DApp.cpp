#include <d3d12.h>
#include <directxmath.h>
#include <dxgi1_6.h>
#include <wrl.h>
#include <utility>
#include <cmath>
#include <exception>
#include <stdexcept>
#include "D3DApp.h"
#include "vertex_shader.h",
#include "pixel_shader.h".

using DirectX::XMFLOAT4X4;
using DirectX::XMFLOAT4;
using DirectX::XMMATRIX;
using namespace DirectX;

struct vs_const_buffer_t {
	XMFLOAT4X4 matWorldViewProj;
	XMFLOAT4X4 matWorldView;
	XMFLOAT4X4 matView;

	XMFLOAT4 colMaterial;
	XMFLOAT4 colLight;
	XMFLOAT4 dirLight;
	XMFLOAT4 padding;
};
static_assert(sizeof(vs_const_buffer_t) == 256);

struct vertex_t {
	FLOAT position[3];
	FLOAT normal_vector[3];
	FLOAT color[4];
};

inline void ThrowIfFailed(HRESULT hr) {
	if (FAILED(hr)) {
		throw std::logic_error("Bad HR");
	}
}

namespace {
	const INT FrameCount = 2;
	using Microsoft::WRL::ComPtr;
	ComPtr<IDXGISwapChain3> swapChain;
	ComPtr<IDXGIFactory7> factory;
	ComPtr<ID3D12Device> device;
	ComPtr<ID3D12Resource> renderTargets[FrameCount];
	ComPtr<ID3D12CommandAllocator> commandAllocator;
	ComPtr<ID3D12CommandQueue> commandQueue;
	
	ComPtr<ID3D12RootSignature> rootSignature;

	typedef ComPtr<ID3D12DescriptorHeap> HeapType;
	HeapType rtvHeap;
	HeapType cbvHeap;

	D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle;
	ComPtr<ID3D12PipelineState> pipelineState;
	ComPtr<ID3D12GraphicsCommandList> commandList;
	UINT rtvDescriptorSize;

	ComPtr<ID3D12Fence> fence;
	UINT frameIndex;
	UINT64 fenceValue;
	HANDLE fenceEvent;

	D3D12_VIEWPORT viewport;

	ComPtr<ID3D12Resource> vertexBuffer;
	D3D12_VERTEX_BUFFER_VIEW vertexBufferView;

	D3D12_DESCRIPTOR_RANGE rootDescRange;
	D3D12_ROOT_PARAMETER rootParameter[1];

	ComPtr<ID3D12Resource> vsConstBuffer;
	
	// Matrices are assigned dynamicly 
	vs_const_buffer_t vsConstBufferData = {
		.colMaterial = {0.4, 1, 0.4, 1},
		.colLight = {1, 1, 1, 1},
		.dirLight = {0.0, 0.0, 1.0, 1},
	};
	UINT8* vsConstBufferPointer;

	RECT rc;

	ComPtr<ID3D12Resource> depthBuffer;
	HeapType depthBufferHeap;

	constexpr size_t VERTEX_SIZE = sizeof(vertex_t) / sizeof(FLOAT);

	constexpr size_t NUM_TRIANGLES = 60;
	constexpr size_t VERTEX_COUNT = NUM_TRIANGLES * 3;

	vertex_t triangle_data[VERTEX_COUNT];

	constexpr size_t VERTEX_BUFFER_SIZE = sizeof(triangle_data);
}

struct SimpleColor {
	FLOAT r, g, b;
};

struct SimpleVertex {
	FLOAT x;
	FLOAT y;
	FLOAT z;
	SimpleColor color;
};

struct Triangle {
	vertex_t t1[3];
};

constexpr Triangle makeSingleTriangle(SimpleVertex p1, SimpleVertex p2, SimpleVertex p3) {
	FLOAT Ax = p2.x - p1.x;
	FLOAT Ay = p2.y - p1.y;
	FLOAT Az = p2.z - p1.z;

	FLOAT Bx = p3.x - p1.x;
	FLOAT By = p3.y - p1.y;
	FLOAT Bz = p3.z - p1.z;

	FLOAT Nx = Ay * Bz - Az * By;
	FLOAT Ny = Az * Bx - Ax * Bz;
	FLOAT Nz = Ax * By - Ay * Bx;


	return {{
		{p1.x, p1.y, p1.z,     Nx, Ny, Nz,    p1.color.r, p1.color.g, p1.color.b, 1.0f},
		{p2.x, p2.y, p2.z,     Nx, Ny, Nz,    p2.color.r, p2.color.g, p2.color.b, 1.0f},
		{p3.x, p3.y, p3.z,     Nx, Ny, Nz,    p3.color.r, p3.color.g, p3.color.b, 1.0f},
	}};
}

constexpr std::pair<Triangle, Triangle> makeTriangle(SimpleVertex p1, SimpleVertex p2, SimpleVertex p3) {
	return {makeSingleTriangle(p1, p2, p3), makeSingleTriangle(p3, p2, p1)};
}

void initTriangleData(size_t triangle_number, Triangle data) {
	size_t idx = triangle_number*3;
	triangle_data[idx + 0] = data.t1[0];
	triangle_data[idx + 1] = data.t1[1];
	triangle_data[idx + 2] = data.t1[2];
}

void initTriangleData() {
	constexpr static SimpleColor green = {0.2f, 1.0f, 0.2f};
	constexpr static SimpleColor white = {1.0f, 1.0f, 1.0f};

	constexpr static size_t segment_count = 3;
	constexpr static FLOAT segments_size[segment_count] = {1.2f, .8f, .5f};

	size_t tr_ind = 0;

	FLOAT base_y = -1.f;
	FLOAT y_scale = 1;
	FLOAT x_scale = 0.5;

	for (size_t l = 0; l < segment_count; l++) {
		FLOAT dx = x_scale * segments_size[l];
		FLOAT dy = y_scale * segments_size[l];
		for (size_t i = 0; i < 10; i++) {

			auto t =  makeTriangle(
				{0, base_y, 0, green},
				{0, base_y+dy, .0, white},
				{std::sinf(2 * 3.14f * i / 10)*dx, base_y, std::cosf(2 * 3.14f * i / 10)*dx, green} 
			);
			initTriangleData(tr_ind++, t.first);
			initTriangleData(tr_ind++, t.second);
			
		}
		base_y  += dy;
	}
	
}

void copyConstBufferToGpu() {
	memcpy(
		vsConstBufferPointer,
		&vsConstBufferData,
		sizeof(vsConstBufferData)
	);
}

void copyTriangleDataToVertexBuffer() {
	UINT8* pVertexDataBegin;
	D3D12_RANGE readRange(0, 0);        // We do not intend to read from this resource on the CPU.
	ThrowIfFailed(vertexBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
	memcpy(pVertexDataBegin, triangle_data, sizeof(triangle_data));
	vertexBuffer->Unmap(0, nullptr);
}

void calcNewMatrix() {
	XMMATRIX wvp_matrix;
	static FLOAT angle = 0.0;
	angle += 0.005;

	XMStoreFloat4x4(
		&vsConstBufferData.matWorldView, 	// zmienna typu vs_const_buffer_t z pkt. 2d
		XMMatrixIdentity()
	);

	wvp_matrix = XMMatrixMultiply(
		XMMatrixRotationY(2.5f * angle),
		XMMatrixRotationX(0)
	);

	XMStoreFloat4x4(
		&vsConstBufferData.matView, 	// zmienna typu vs_const_buffer_t z pkt. 2d
		wvp_matrix
	);

	wvp_matrix = XMMatrixMultiply(
		wvp_matrix,
		XMMatrixTranslation(0.0f, 0.0f, 4.0f)
	);

	wvp_matrix = XMMatrixMultiply(
		wvp_matrix, 
		XMMatrixPerspectiveFovLH(
			45.0f, viewport.Width / viewport.Height, 1.0f, 100.0f
		)
	);
	wvp_matrix = XMMatrixTranspose(wvp_matrix);
	XMStoreFloat4x4(
		&vsConstBufferData.matWorldViewProj, 	// zmienna typu vs_const_buffer_t z pkt. 2d
		wvp_matrix
	);

	copyConstBufferToGpu();
}

namespace DXInitAux {
	void inline createHeap(const D3D12_DESCRIPTOR_HEAP_DESC& desc, HeapType& heap) {
		ThrowIfFailed(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heap)));
		if (heap == nullptr) {
			throw std::logic_error("Heap is a nullptr");
		}
	}

	void inline createBasicCommittedResource(
		const D3D12_HEAP_PROPERTIES *pHeapProperties,
		const D3D12_RESOURCE_DESC *pDesc,
		ComPtr<ID3D12Resource>& resource) {

		ThrowIfFailed(device->CreateCommittedResource(
			pHeapProperties,
			D3D12_HEAP_FLAG_NONE,
			pDesc,
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS(&resource)
		));
	}

	void initDepthBuffer() {
		D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {
			.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
			.NumDescriptors = 1,
			.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
			.NodeMask = 0,
		};
		D3D12_HEAP_PROPERTIES heapProp = {
			.Type = D3D12_HEAP_TYPE_DEFAULT,
			.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
			.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,      
			.CreationNodeMask = 1,
			.VisibleNodeMask = 1,
		};
		D3D12_RESOURCE_DESC resDesc = {
			.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
			.Alignment = 0,
			.Width = UINT64(rc.right - rc.left),
			.Height = UINT64(rc.bottom - rc.top),
			.DepthOrArraySize = 1,
			.MipLevels = 0,
			.Format = DXGI_FORMAT_D32_FLOAT,
			.SampleDesc = {.Count = 1, .Quality = 0 },
			.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN,
			.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
		};
		// D3D12_CLEAR_VALUE clearValue = {
		// 	.Format = DXGI_FORMAT_D32_FLOAT,
		// 	.DepthStencil = { .Depth = 1.0f, .Stencil = 0 }
		// };
		D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilViewDesc = {
			.Format = DXGI_FORMAT_D32_FLOAT,
			.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D,
			.Flags = D3D12_DSV_FLAG_NONE,
			.Texture2D = {}
		};

		createHeap(heapDesc, depthBufferHeap);

		DXInitAux::createBasicCommittedResource(&heapProp, &resDesc, depthBuffer);
		
		device->CreateDepthStencilView(
			depthBuffer.Get(),
			&depthStencilViewDesc,
			depthBufferHeap->GetCPUDescriptorHandleForHeapStart()
		);
	}

	void initDeviceAndFactory() {
		ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));

		ThrowIfFailed(D3D12CreateDevice(
			nullptr,
			D3D_FEATURE_LEVEL_12_0,
			IID_PPV_ARGS(&device)
		));
	}

	void initViewPort() {
		viewport = {
			.TopLeftX = 0.f,
			.TopLeftY = 0.f,
			.Width = FLOAT(rc.right - rc.left),
			.Height = FLOAT(rc.bottom - rc.top),
			.MinDepth = 0.0f,
			.MaxDepth = 1.0f
		};
	}

	void initSwapChain(HWND hwnd) {
			DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {
			.Width = 0,
			.Height = 0,
			.Format = DXGI_FORMAT_R8G8B8A8_UNORM,
			.SampleDesc = { .Count = 1 },
			.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT,
			.BufferCount = FrameCount,
			.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD,
		};

		ComPtr<IDXGISwapChain1> tempSwapChain;
		ThrowIfFailed(factory->CreateSwapChainForHwnd(
			commandQueue.Get(),        // Swap chain needs the queue so that it can force a flush on it.
			hwnd,
			&swapChainDesc,
			nullptr,
			nullptr,
			&tempSwapChain
		));
		ThrowIfFailed(factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER));
		ThrowIfFailed(tempSwapChain.As(&swapChain));

		frameIndex = swapChain->GetCurrentBackBufferIndex();
	}

	void initVertexBuffer() {
		// Note: using upload heaps to transfer static data like vert buffers is not 
		// recommended. Every time the GPU needs it, the upload heap will be marshalled 
		// over. Please read up on Default Heap usage. An upload heap is used here for 
		// code simplicity and because there are very few verts to actually transfer.
		D3D12_HEAP_PROPERTIES heapProps = {
			.Type = D3D12_HEAP_TYPE_UPLOAD,
			.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
			.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
			.CreationNodeMask = 1,
			.VisibleNodeMask = 1,
		};

		D3D12_RESOURCE_DESC desc = {
			.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
			.Alignment = 0,
			.Width = VERTEX_BUFFER_SIZE,
			.Height = 1,
			.DepthOrArraySize = 1,
			.MipLevels = 1,
			.Format = DXGI_FORMAT_UNKNOWN,
			.SampleDesc = {.Count = 1, .Quality = 0 },
			.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
			.Flags = D3D12_RESOURCE_FLAG_NONE,
		};
   
		DXInitAux::createBasicCommittedResource(&heapProps, &desc, vertexBuffer);
		copyTriangleDataToVertexBuffer();

		// Initialize the vertex buffer view.
		vertexBufferView.BufferLocation = vertexBuffer->GetGPUVirtualAddress();
		vertexBufferView.StrideInBytes = sizeof(vertex_t);
		vertexBufferView.SizeInBytes = VERTEX_BUFFER_SIZE;
	}

	void initCommandQueue() {
		D3D12_COMMAND_QUEUE_DESC queueDesc = {
			.Type = D3D12_COMMAND_LIST_TYPE_DIRECT,
			.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE,
		};
		ThrowIfFailed(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&commandQueue)));
	}

	void initCBVRTVHeaps() {
		D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {
			.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
			.NumDescriptors = FrameCount,
			.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
		};
		D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc = {
			.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
			.NumDescriptors = 1,
			.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
			.NodeMask = 0  
		};

		DXInitAux::createHeap(rtvHeapDesc, rtvHeap);
		DXInitAux::createHeap(cbvHeapDesc, cbvHeap);
	}

	void initPipelineState() {
		D3D12_INPUT_ELEMENT_DESC inputElementDescs[] = {
			{
				.SemanticName = "POSITION",
				.SemanticIndex = 0,
				.Format = DXGI_FORMAT_R32G32B32_FLOAT,
				.InputSlot = 0,
				.AlignedByteOffset = D3D12_APPEND_ALIGNED_ELEMENT,
				.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
				.InstanceDataStepRate = 0
			},
			{
				.SemanticName = "NORMAL",
				.SemanticIndex = 0,
				.Format = DXGI_FORMAT_R32G32B32_FLOAT,
				.InputSlot = 0,
				.AlignedByteOffset = D3D12_APPEND_ALIGNED_ELEMENT,
				.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
				.InstanceDataStepRate = 0
			},
			{
				.SemanticName = "COLOR",
				.SemanticIndex = 0,
				.Format = DXGI_FORMAT_R32G32B32A32_FLOAT,
				.InputSlot = 0,
				.AlignedByteOffset = D3D12_APPEND_ALIGNED_ELEMENT,
				.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
				.InstanceDataStepRate = 0
			}
		};

		D3D12_BLEND_DESC blendDesc = {
			.AlphaToCoverageEnable = FALSE,
			.IndependentBlendEnable = FALSE,
			.RenderTarget = {
				{
				.BlendEnable = FALSE,
				.LogicOpEnable = FALSE,
				.SrcBlend = D3D12_BLEND_ONE,
				.DestBlend = D3D12_BLEND_ZERO,
				.BlendOp = D3D12_BLEND_OP_ADD,
				.SrcBlendAlpha = D3D12_BLEND_ONE,
				.DestBlendAlpha = D3D12_BLEND_ZERO,
				.BlendOpAlpha = D3D12_BLEND_OP_ADD,
				.LogicOp = D3D12_LOGIC_OP_NOOP,
				.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL
				}
			}
		};

		D3D12_RASTERIZER_DESC rasterizerDesc = {
			.FillMode = D3D12_FILL_MODE_SOLID,
			.CullMode = D3D12_CULL_MODE_BACK,
			.FrontCounterClockwise = FALSE,
			.DepthBias = D3D12_DEFAULT_DEPTH_BIAS,
			.DepthBiasClamp = D3D12_DEFAULT_DEPTH_BIAS_CLAMP,
			.SlopeScaledDepthBias = D3D12_DEFAULT_SLOPE_SCALED_DEPTH_BIAS,
			.DepthClipEnable = TRUE,
			.MultisampleEnable = FALSE,
			.AntialiasedLineEnable = FALSE,
			.ForcedSampleCount = 0,
			.ConservativeRaster = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF
		};

		D3D12_DEPTH_STENCIL_DESC depthStencilDesc = {
			.DepthEnable = TRUE,
			.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL,
			.DepthFunc = D3D12_COMPARISON_FUNC_LESS,
			.StencilEnable = FALSE,
			.StencilReadMask = D3D12_DEFAULT_STENCIL_READ_MASK,
			.StencilWriteMask = D3D12_DEFAULT_STENCIL_READ_MASK,
			.FrontFace = {
				.StencilFailOp = D3D12_STENCIL_OP_KEEP,
				.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP,
				.StencilPassOp = D3D12_STENCIL_OP_KEEP,
				.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS
			},
			.BackFace = {
				.StencilFailOp = D3D12_STENCIL_OP_KEEP,
				.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP,
				.StencilPassOp = D3D12_STENCIL_OP_KEEP,
				.StencilFunc = D3D12_COMPARISON_FUNC_ALWAYS
			}
		};

		// D3D12_GRAPHICS_PIPELINE_STATE_DESC pipelineStateDesc = {
		// 	.DepthStencilState = depthStencilDesc,
		// 	.DSVFormat = DXGI_FORMAT_D32_FLOAT,
		// };

		D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {
			.pRootSignature = rootSignature.Get(),
			.VS = { vs_main, sizeof(vs_main) },
			.PS = { ps_main, sizeof(ps_main) },
			.BlendState = blendDesc,
			.SampleMask = UINT_MAX,
			.RasterizerState = rasterizerDesc,
			.DepthStencilState = depthStencilDesc,
			.InputLayout = { inputElementDescs, _countof(inputElementDescs) },
			.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
			.NumRenderTargets = 1,
			.DSVFormat = DXGI_FORMAT_D32_FLOAT,
			.SampleDesc = {.Count = 1},
		};
		psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;

		ThrowIfFailed(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pipelineState)));
	}

	void initRootSignature() {
		rootDescRange = {
			.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV,
			.NumDescriptors = 1,
			.BaseShaderRegister = 0,
			.RegisterSpace = 0,
			.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND,
		};

		rootParameter[0] = {
			.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
			.DescriptorTable = { 1, &rootDescRange},	// adr. rekordu poprzedniego typu
			.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX,
		};

		D3D12_ROOT_SIGNATURE_DESC rootSignatureDesc = {
			.NumParameters = _countof(rootParameter),
			.pParameters = rootParameter,
			.NumStaticSamplers = 0,
			.pStaticSamplers = nullptr,
			.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
					D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
					D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
					D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
					D3D12_ROOT_SIGNATURE_FLAG_DENY_PIXEL_SHADER_ROOT_ACCESS,
		};

		ComPtr<ID3DBlob> signature;
		ComPtr<ID3DBlob> error;
		ThrowIfFailed(D3D12SerializeRootSignature(
			&rootSignatureDesc, 
			D3D_ROOT_SIGNATURE_VERSION_1,
			&signature, &error
		));
		ThrowIfFailed(device->CreateRootSignature(
			0,
			signature->GetBufferPointer(), signature->GetBufferSize(),
			IID_PPV_ARGS(&rootSignature)
		));
	}

	void initVsConstBufferResourceAndView() {
		D3D12_HEAP_PROPERTIES vsHeapTypeProp = {
			.Type = D3D12_HEAP_TYPE_UPLOAD,
			.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
			.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN,
			.CreationNodeMask = 1,
			.VisibleNodeMask = 1,
		};
		D3D12_RESOURCE_DESC vsHeapResourceDesc =  {
			.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
			.Alignment = 0,
			.Width = sizeof(vs_const_buffer_t),
			.Height = 1,
			.DepthOrArraySize = 1,
			.MipLevels = 1,
			.Format = DXGI_FORMAT_UNKNOWN,
			.SampleDesc = { .Count = 1, .Quality = 0 },
			.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
			.Flags = D3D12_RESOURCE_FLAG_NONE,
		};

		DXInitAux::createBasicCommittedResource(&vsHeapTypeProp, &vsHeapResourceDesc, vsConstBuffer);

		D3D12_CONSTANT_BUFFER_VIEW_DESC vbViewDesc = {
			.BufferLocation = vsConstBuffer->GetGPUVirtualAddress(),
			.SizeInBytes = sizeof(vs_const_buffer_t),
		};
		device->CreateConstantBufferView(&vbViewDesc, cbvHeap->GetCPUDescriptorHandleForHeapStart());

	}

	void initCommandAllocatorAndList() {
		rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
		rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
		for (UINT n = 0; n < FrameCount; n++) {
			ThrowIfFailed(swapChain->GetBuffer(n, IID_PPV_ARGS(&renderTargets[n])));
			device->CreateRenderTargetView(renderTargets[n].Get(), nullptr, rtvHandle);
			rtvHandle.ptr += rtvDescriptorSize;
		}

		ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&commandAllocator)));
		
		ThrowIfFailed(device->CreateCommandList(
			0, D3D12_COMMAND_LIST_TYPE_DIRECT, 
			commandAllocator.Get(), pipelineState.Get(), 
			IID_PPV_ARGS(&commandList)
		));
		ThrowIfFailed(commandList->Close());
	}

}

void PopulateCommandList(HWND hwnd) {
	ThrowIfFailed(commandAllocator->Reset());
	ThrowIfFailed(commandList->Reset(commandAllocator.Get(), pipelineState.Get()));

	commandList->SetGraphicsRootSignature(rootSignature.Get());
	
	ID3D12DescriptorHeap* ppHeaps[] = { cbvHeap.Get() };
	commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);
	
	commandList->SetGraphicsRootDescriptorTable(0, cbvHeap->GetGPUDescriptorHandleForHeapStart());
	
	commandList->RSSetViewports(1, &viewport);
	commandList->RSSetScissorRects(1, &rc);

	D3D12_RESOURCE_BARRIER barrier = {
		.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
		.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
		.Transition = {
			.pResource = renderTargets[frameIndex].Get(),
			.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
			.StateBefore = D3D12_RESOURCE_STATE_PRESENT,
			.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET
		}
	};
	commandList->ResourceBarrier(1, &barrier);

	auto rtvHandleHeapStart = rtvHeap->GetCPUDescriptorHandleForHeapStart();
	rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
	
	rtvHandleHeapStart.ptr += frameIndex * rtvDescriptorSize;
	
	D3D12_CPU_DESCRIPTOR_HANDLE cpudesc = depthBufferHeap->GetCPUDescriptorHandleForHeapStart();
	
	commandList->OMSetRenderTargets(
		1, &rtvHandleHeapStart,
		FALSE, 
		&cpudesc
	);

	const float clearColor[] = { 0.0f, 0.8f, 0.8f, 1.0f };
	commandList->ClearRenderTargetView(rtvHandleHeapStart, clearColor, 0, nullptr);
	commandList->ClearDepthStencilView(
		cpudesc,
		D3D12_CLEAR_FLAG_DEPTH , 1.0f, 0, 0, nullptr
	);

	commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
	commandList->IASetVertexBuffers(0, 1, &vertexBufferView);
	commandList->DrawInstanced(VERTEX_COUNT, 1, 0, 0);

	D3D12_RESOURCE_BARRIER barrier2 = {
		.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
		.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE,
		.Transition = {
			.pResource = renderTargets[frameIndex].Get(),
			.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
			.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET,
			.StateAfter = D3D12_RESOURCE_STATE_PRESENT
		}
	};
	commandList->ResourceBarrier(1, &barrier2);

	ThrowIfFailed(commandList->Close());
}

void WaitForPreviousFrame(HWND hwnd) {
	// WAITING FOR THE FRAME TO COMPLETE BEFORE CONTINUING IS NOT BEST PRACTICE.
	// This is code implemented as such for simplicity. More advanced samples 
	// illustrate how to use fences for efficient resource usage.

	// Signal and increment the fence value.
	const UINT64 fenceVal = fenceValue;
	ThrowIfFailed(commandQueue->Signal(fence.Get(), fenceVal));
	fenceValue++;

	// Wait until the previous frame is finished.
	if (fence->GetCompletedValue() < fenceVal) {
		ThrowIfFailed(fence->SetEventOnCompletion(fenceVal, fenceEvent));
		WaitForSingleObject(fenceEvent, INFINITE);
	}

	frameIndex = swapChain->GetCurrentBackBufferIndex();
}

void InitDirect3D(HWND hwnd) {
	
	initTriangleData();
	if (GetClientRect(hwnd, &rc) == 0) {
		throw std::logic_error("GetClientRect failed");
	}
	DXInitAux::initDeviceAndFactory();
	DXInitAux::initViewPort();
	DXInitAux::initCommandQueue();
	DXInitAux::initSwapChain(hwnd);
	DXInitAux::initCBVRTVHeaps();
	DXInitAux::initCommandAllocatorAndList();
	
	DXInitAux::initDepthBuffer();
	DXInitAux::initVsConstBufferResourceAndView();

	DXInitAux::initRootSignature();
	DXInitAux::initPipelineState();
	DXInitAux::initVertexBuffer();

	ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
	fenceValue = 1;

	// Create an event handle to use for frame synchronization.
	fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
	if (fenceEvent == nullptr) {
		ThrowIfFailed(HRESULT_FROM_WIN32(GetLastError()));
	}

	D3D12_RANGE constBufferDataRange = {0, 0};
	ThrowIfFailed(vsConstBuffer->Map(0, &constBufferDataRange, reinterpret_cast<void**>(&vsConstBufferPointer)));
	copyConstBufferToGpu();

	// Wait for the command list to execute; we are reusing the same command 
	// list in our main loop but for now, we just want to wait for setup to 
	// complete before continuing.
	WaitForPreviousFrame(hwnd);
}

void OnUpdate(HWND hwnd) {
	calcNewMatrix();
}

void OnRender(HWND hwnd) {
	// Record all the commands we need to render the scene into the command list.
	PopulateCommandList(hwnd);

	// Execute the command list.
	ID3D12CommandList* ppCommandLists[] = { commandList.Get() };
	commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

	// Present the frame.
	ThrowIfFailed(swapChain->Present(1, 0));
	WaitForPreviousFrame(hwnd);
}

void OnDestroy(HWND hwnd) {
	// Wait for the GPU to be done with all resources.
	WaitForPreviousFrame(hwnd);
	
	CloseHandle(fenceEvent);
}